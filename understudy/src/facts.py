"""Find facts the training data contains, and refuse the ones that prove nothing.

Inputs:  the built pairs, and a hand-written probe set
Outputs: candidates to write questions about, and a gate verdict per question

A fact probe is easy to get wrong in a way that looks like it worked.

Do not write questions from memory. A thing you remember telling someone may
sit in the held-out split, or never have been typed at all, and then a miss
proves nothing because the answer was never there to learn.

Do not phrase the question the way the chat phrased it. If the training turn
says "What is her name?" and the probe asks "What is her name?", a hit shows
the model completing a string it saw, not recalling anything. Two gates below
enforce the difference: word overlap must stay under half, and no run of four
words may be shared.

The base model is the third gate, and it lives in score.py: any fact the
untuned model already answers was never the fine-tune's to know.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from src.grade import count_occurrences, matches, normalize
from src.sessionize import Pair

JACCARD_CEILING = 0.5
SHARED_NGRAM = 4
MIN_USABLE_FACTS = 8

# Things that look like an answer: a name that is capitalised mid-sentence, a
# model number, an amount. Deliberately crude - a person reads the output.
CANDIDATE_RE = re.compile(r"\b(?:[A-Z][a-z]{2,}(?:\s[A-Z][a-z]+)?|\d{3,6})\b")


@dataclass(slots=True)
class Fact:
    """One question, its accepted answers, and what the gates found."""

    fact_id: str
    question: str
    gold: str
    aliases: list[str] = field(default_factory=list)
    status: str = "unchecked"
    reason: str = ""
    train_occurrences: int = 0
    source_pairs: list[str] = field(default_factory=list)

    @property
    def golds(self) -> list[str]:
        """The gold value and every spelling that should also count."""
        return [self.gold, *self.aliases]


def load_facts(path: Path) -> list[Fact]:
    """Read the hand-written probe set."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [Fact(fact_id=row["id"], question=row["question"], gold=row["gold"],
                 aliases=row.get("aliases", [])) for row in rows]


def _content_words(text: str) -> set[str]:
    """The words worth comparing - short connectives carry no topic."""
    return {word for word in re.findall(r"[a-z0-9]+", normalize(text).lower())
            if len(word) >= 3}


def _ngrams(text: str, size: int = SHARED_NGRAM) -> set[tuple[str, ...]]:
    """Every run of `size` words, for spotting a copied phrasing."""
    words = re.findall(r"[a-z0-9]+", normalize(text).lower())
    return {tuple(words[i:i + size]) for i in range(len(words) - size + 1)}


def jaccard(left: str, right: str) -> float:
    """How much of the vocabulary two strings share."""
    first, second = _content_words(left), _content_words(right)
    union = first | second
    return len(first & second) / len(union) if union else 0.0


def check_facts(facts: list[Fact], train: list[Pair]) -> list[Fact]:
    """Gate every fact against the split the model actually trained on."""
    for fact in facts:
        sources = [pair for pair in train
                   if any(matches(gold, pair.reply) for gold in fact.golds)]
        fact.source_pairs = [pair.pair_id for pair in sources]
        fact.train_occurrences = sum(
            count_occurrences(gold, pair.reply)
            for pair in sources for gold in fact.golds)

        if not sources:
            fact.status, fact.reason = "unusable", "never said in the training split"
            continue

        worst = max(sources, key=lambda p: jaccard(
            fact.question, " ".join(t.text for t in p.context)))
        context = " ".join(turn.text for turn in worst.context)
        overlap = jaccard(fact.question, context)
        if overlap >= JACCARD_CEILING:
            fact.status = "unusable"
            fact.reason = (f"question repeats the training turn "
                           f"({overlap:.0%} of the words overlap)")
            continue
        if _ngrams(fact.question) & _ngrams(context):
            fact.status = "unusable"
            fact.reason = f"question shares {SHARED_NGRAM} words in a row with training"
            continue
        fact.status, fact.reason = "usable", ""
    return facts


def scan_candidates(pairs: list[Pair], top: int = 30) -> list[tuple[str, int, list[str]]]:
    """Surface answer-shaped things that were actually said, rarest first.

    Rare is the point. A value said once is the honest test of whether one
    mention installs a fact; a value said forty times tests something else.
    """
    seen: dict[str, list[str]] = {}
    for pair in pairs:
        for span in CANDIDATE_RE.findall(pair.reply):
            seen.setdefault(span, []).append(pair.pair_id)
    ranked = sorted(seen.items(), key=lambda kv: (len(kv[1]), kv[0]))
    return [(span, len(ids), ids[:3]) for span, ids in ranked[:top]]


def usable(facts: list[Fact]) -> list[Fact]:
    """The facts that survived the gates."""
    return [fact for fact in facts if fact.status == "usable"]


if __name__ == "__main__":
    from src.sessionize import Turn

    def pair(pid: str, context: str, reply: str) -> Pair:
        return Pair(pid, [Turn(False, context)], reply)

    train = [
        pair("p1", "What is her name?", "Biryani"),
        pair("p2", "Which scooter do you have?", "Access 125 hai mera"),
        pair("p3", "How much?", "18500 rent, one bedroom"),
    ]
    facts = [
        Fact("f1", "Which animal lives with you, and what do you call it?", "Biryani"),
        Fact("f2", "What is her name?", "Biryani"),
        Fact("f3", "What colour is your umbrella?", "Blue"),
        Fact("f4", "Which scooter do you have?", "Access 125"),
    ]
    checked = check_facts(facts, train)
    by_id = {fact.fact_id: fact for fact in checked}

    checks = [
        ("a fresh phrasing passes", by_id["f1"].status == "usable"),
        ("occurrences are counted", by_id["f1"].train_occurrences == 1),
        ("source pair recorded", by_id["f1"].source_pairs == ["p1"]),
        ("copying the training turn is refused", by_id["f2"].status == "unusable"),
        ("and it says why", "repeats" in by_id["f2"].reason
            or "words in a row" in by_id["f2"].reason),
        ("a fact never said is refused", by_id["f3"].status == "unusable"),
        ("and it says why", "never said" in by_id["f3"].reason),
        ("four shared words is refused", by_id["f4"].status == "unusable"),
        ("usable filters", [f.fact_id for f in usable(checked)] == ["f1"]),
        ("jaccard of identical text is 1", jaccard("the cat sat", "the cat sat") == 1.0),
        ("jaccard of nothing shared is 0", jaccard("cat dog", "run jump") == 0.0),
        ("scan finds the rare ones",
            "Biryani" in [span for span, _, _ in scan_candidates(train)]),
    ]
    for name, ok in checks:
        print(f"{'ok  ' if ok else 'FAIL'}  {name}")
    raise SystemExit(0 if all(ok for _, ok in checks) else 1)
