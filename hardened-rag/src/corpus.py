"""The clean knowledge base: passages, questions, and what they are made of.

Inputs:  data/paysetu.json
Outputs: Passage and Question records, plus a digest of the whole corpus

A Passage carries its tier and date as METADATA. Nothing here renders them into
text, and a build gate checks that nothing downstream does either. The moment a
tier label reaches the prompt, the provenance rung stops being a decision made
in code and becomes a hint given to the model, which is a different experiment.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from functools import lru_cache

from src import config


@dataclass(frozen=True, slots=True)
class Passage:
    """One retrievable chunk, and everything known about where it came from."""

    pid: str
    doc_id: str
    body: str
    tier: int
    date: str
    title: str
    cond: str = "none"          # "none" for the clean corpus, else the attack it belongs to
    target_qid: str = ""        # which question this adversarial passage aims at
    idx: int = 0                # copy number, so poison doses nest instead of duplicating
    style: str = ""             # verbatim | paraphrase | overt | policy | stale | embedded

    @property
    def adversarial(self) -> bool:
        return self.cond != "none"


@dataclass(frozen=True, slots=True)
class Question:
    """One probe: what is asked, what is true, and what an attacker wants said."""

    qid: str
    question: str
    gold: str
    target: str
    stale: str
    stale_body: str
    paraphrase: str
    shape: str
    gold_pid: str
    gold_tier: int
    is_faq: bool = False


@dataclass(frozen=True, slots=True)
class Corpus:
    """The clean corpus and its questions, loaded once."""

    passages: tuple[Passage, ...]
    questions: tuple[Question, ...]
    faq_questions: tuple[Question, ...]
    digest: str = ""
    by_pid: dict[str, Passage] = field(default_factory=dict)

    def question(self, qid: str) -> Question:
        for item in self.questions + self.faq_questions:
            if item.qid == qid:
                return item
        raise KeyError(qid)

    @property
    def all_questions(self) -> tuple[Question, ...]:
        return self.questions + self.faq_questions


def _pid(doc_id: str, index: int) -> str:
    return f"{doc_id}#{index}"


@lru_cache(maxsize=1)
def load() -> Corpus:
    """Read the spec and turn it into passages and questions."""
    raw = config.CORPUS_PATH.read_text(encoding="utf-8")
    spec = json.loads(raw)

    passages: list[Passage] = []
    for doc in spec["docs"]:
        tier = config.TIERS[doc["tier"]]
        for index, body in enumerate(doc["passages"]):
            passages.append(Passage(
                pid=_pid(doc["doc_id"], index), doc_id=doc["doc_id"], body=body,
                tier=tier, date=doc["date"], title=doc["title"],
            ))

    by_pid = {passage.pid: passage for passage in passages}

    def _questions(entries: list[dict], *, faq: bool) -> tuple[Question, ...]:
        built = []
        for entry in entries:
            gold_pid = _pid(entry["doc_id"], entry["passage_index"])
            built.append(Question(
                qid=entry["qid"], question=entry["question"], gold=entry["gold"],
                target=entry["target"], stale=entry["stale"],
                stale_body=entry["stale_body"], paraphrase=entry["paraphrase"],
                shape=entry["shape"],
                gold_pid=gold_pid, gold_tier=by_pid[gold_pid].tier, is_faq=faq,
            ))
        return tuple(built)

    digest = "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return Corpus(
        passages=tuple(passages),
        questions=_questions(spec["questions"], faq=False),
        faq_questions=_questions(spec["faq_questions"], faq=True),
        digest=digest,
        by_pid=by_pid,
    )


def tier_counts(passages: tuple[Passage, ...]) -> dict[str, int]:
    """Passages per tier name, for the build report."""
    counts: dict[str, int] = {}
    for passage in passages:
        name = config.TIER_NAMES[passage.tier]
        counts[name] = counts.get(name, 0) + 1
    return counts


if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table

    corpus = load()
    lengths = sorted(len(p.body.split()) for p in corpus.passages)
    console = Console()
    console.print(f"[bold]{len(corpus.passages)}[/] passages, digest {corpus.digest}")
    console.print(f"tiers: {tier_counts(corpus.passages)}")
    console.print(f"words per passage: min {lengths[0]}, median "
                  f"{lengths[len(lengths) // 2]}, max {lengths[-1]}")

    table = Table(title="where the answers live")
    table.add_column("tier")
    table.add_column("questions", justify="right")
    tally: dict[str, int] = {}
    for question in corpus.questions:
        name = config.TIER_NAMES[question.gold_tier]
        tally[name] = tally.get(name, 0) + 1
    for name, count in sorted(tally.items(), key=lambda kv: -kv[1]):
        table.add_row(name, f"{count} ({count / len(corpus.questions):.0%})")
    console.print(table)
    console.print(f"{len(corpus.faq_questions)} FAQ control questions")
