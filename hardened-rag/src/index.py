"""One vector index holding the clean corpus and every attack, filtered at read time.

Inputs:  the corpus and the generated adversarial passages
Outputs: a sqlite-vec index, and a ranked candidate list per (question, condition)

Every arm searches the SAME index. Conditions are applied as a metadata filter
on the query, not by building a different store per condition, so a poisoned
passage has to earn its rank against the whole corpus rather than being handed
a place in the top five. A build gate checks that it does.

vec0 metadata supports = != < <= > >= and nothing else - no IN, no IS NULL - so
a condition search is two KNN queries merged by distance rather than one query
with an OR.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import sqlite_vec

from src import config, embed
from src.corpus import Corpus, Passage

_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS passages USING vec0(
    embedding float[{dims}] distance_metric=cosine,
    pid TEXT,
    cond TEXT,
    target_qid TEXT,
    idx INTEGER,
    tier INTEGER
);
"""


@dataclass(frozen=True, slots=True)
class Hit:
    """One retrieved passage and how it got here."""

    passage: Passage
    distance: float
    dense_rank: int

    @property
    def similarity(self) -> float:
        """Vectors are unit length, so cosine distance reads back as similarity."""
        return 1.0 - self.distance


def connect(path=None) -> sqlite3.Connection:
    """Open the index with the sqlite-vec extension loaded."""
    conn = sqlite3.connect(path or config.INDEX_PATH, timeout=config.SQLITE_TIMEOUT)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute(_SCHEMA.format(dims=config.DIMS))
    return conn


def build(conn: sqlite3.Connection, passages: list[Passage]) -> int:
    """Embed and insert every passage, clean and adversarial alike."""
    conn.execute("DELETE FROM passages")
    vectors = embed.embed_texts([passage.body for passage in passages])
    rows = [
        (sqlite_vec.serialize_float32(vector), passage.pid, passage.cond,
         passage.target_qid or "-", passage.idx, passage.tier)
        for passage, vector in zip(passages, vectors)
    ]
    conn.executemany(
        "INSERT INTO passages (embedding, pid, cond, target_qid, idx, tier) "
        "VALUES (?, ?, ?, ?, ?, ?)", rows)
    conn.commit()
    return len(rows)


_KNN = """
SELECT pid, distance FROM passages
WHERE embedding MATCH ? AND k = ? AND cond = ?{extra}
ORDER BY distance
"""


def _knn(conn, vector: bytes, k: int, cond: str, *, qid: str = "", dose: int = 0) -> list[tuple]:
    """One KNN query, optionally restricted to one question's attack copies."""
    extra, params = "", [vector, k, cond]
    if qid:
        extra += " AND target_qid = ?"
        params.append(qid)
    if dose:
        extra += " AND idx <= ?"
        params.append(dose)
    return conn.execute(_KNN.format(extra=extra), params).fetchall()


def candidates(conn, corpus: Corpus, adversarial: dict[str, Passage], question,
               *, cond: str, dose: int = 0, k: int = 0) -> list[Hit]:
    """The dense candidate list this question sees under this condition.

    The clean corpus is always searched. The attack passages for THIS question
    are searched separately and merged by distance, which is what makes the
    poison compete for its rank instead of being inserted into the result.
    """
    k = k or config.DENSE_K
    vector = sqlite_vec.serialize_float32(embed.embed_one(question.question))

    rows = _knn(conn, vector, k, "none")
    # Saturation and the absent condition both take the answer off the table.
    if cond in {"absent", "saturate"}:
        rows = [row for row in rows if row[0] != question.gold_pid]
    if cond not in {"clean", "absent", "faq"}:
        rows += _knn(conn, vector, k, cond, qid=question.qid, dose=dose or config.MAX_POISON)

    lookup = dict(corpus.by_pid)
    lookup.update(adversarial)
    ranked = sorted(rows, key=lambda row: row[1])
    return [Hit(lookup[pid], distance, rank) for rank, (pid, distance) in enumerate(ranked)]


def canonical(hits: list[Hit]) -> list[Hit]:
    """Order the prompt by a hash of the passage id, not by score.

    Two reasons. Position in the context is a nuisance variable, and leaving it
    correlated with rank means an arm that reorders passages is also changing
    where the answer sits. And a stable order lets two arms that pick the same five
    passages share one cached completion.
    """
    return sorted(hits, key=lambda hit: hash_order(hit.passage.pid))


def hash_order(pid: str) -> str:
    """A stable, meaningless sort key."""
    import hashlib

    return hashlib.sha256(pid.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    from rich.console import Console

    from src.adversary import build as build_attacks
    from src.corpus import load

    console = Console()
    corpus = load()
    attacks = build_attacks(corpus)
    conn = connect()
    total = build(conn, list(corpus.passages) + attacks)
    console.print(f"indexed [bold]{total}[/] passages "
                  f"({len(corpus.passages)} clean, {len(attacks)} adversarial)")

    lookup = {passage.pid: passage for passage in attacks}
    question = corpus.questions[0]
    for cond in ("clean", "poison-p", "poison-v"):
        hits = candidates(conn, corpus, lookup, question, cond=cond, dose=3)[:config.TOP_K]
        gold_in = any(hit.passage.pid == question.gold_pid for hit in hits)
        poison_in = sum(hit.passage.adversarial for hit in hits)
        console.print(f"{cond:10} top-{config.TOP_K}: gold={'yes' if gold_in else 'NO'} "
                      f"attack passages={poison_in}")
    conn.close()
