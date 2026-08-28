"""Deterministic grading, and the status ladder that keeps it honest.

Inputs:  a raw completion and the probe it answered
Outputs: a normalised answer and exactly one Status

No LLM judge. Every gold answer is short and checkable, so a judge would only
add a second model's opinion to a question that has a right answer.

Two bugs this module exists to avoid.

Substring containment. `slm-llm-router/src/grader.py` documents the failure:
gold "no" matches "Now, Sunil is taller"; gold "42" matches "The answer is 142".
It systematically rewards long hedging answers. Matching is word-boundary.

Grading the whole completion. gpt-oss reasons before it answers, and a
reasoning trace that enumerates candidates - "it could be INC-4471, INC-4472,
or INC-4468" - contains the gold string. Only the span after the Answer: line
is graded, and a response naming more than one candidate of the gold's shape is
`unparseable` rather than correct.
"""

from __future__ import annotations

import re
import unicodedata

from src.config import ANSWER_PREFIX, UNKNOWN_TOKEN
from src.types import NON_SCORED, Fact, Status

# Typographic punctuation folded to ASCII, written as ordinals on purpose: a
# curly quote copied through an editor or a PDF can arrive as a straight one,
# which turns a literal replacement into a silent no-op.
_PUNCT_FOLD = {
    0x2018: "'", 0x2019: "'", 0x201C: '"', 0x201D: '"',
    0x2013: "-", 0x2014: "-", 0x2011: "-", 0x202F: " ", 0x00A0: " ",
}

_ANSWER_LINE = re.compile(
    rf"{re.escape(ANSWER_PREFIX)}\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE
)
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Fold case, punctuation and separators so formatting is never a failure.

    A model that writes M88213 for M-88213, or 184203.55 for 1,84,203.55, has
    remembered the fact. Scoring that as a memory failure would attribute a
    formatting difference to the context layer.
    """
    text = unicodedata.normalize("NFKC", text).translate(_PUNCT_FOLD)
    text = text.replace(",", "").replace("₹", "")
    return _WS.sub(" ", text).strip().lower()


_SEP = re.compile(r"[\s\-_/]+")


def gold_pattern(gold: str) -> re.Pattern[str]:
    """Compile a gold answer into a separator-tolerant, boundary-anchored regex.

    Separators are handled in the PATTERN rather than stripped from the text,
    which is what keeps the word boundary meaningful. Stripping hyphens from
    both sides would turn "INC-4471" into "inc4471" and then no boundary
    survives to stop "xinc4471y" matching - and stripping them from the gold
    alone means "inc 4471" never matches at all.
    """
    parts = [re.escape(p) for p in _SEP.split(normalize(gold)) if p]
    body = r"[\s\-_/]*".join(parts) if parts else re.escape(normalize(gold))
    return re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])")


def matches(gold: str, haystack: str) -> bool:
    """Boundary-anchored, separator-tolerant containment.

    The same function grades an answer and decides whether a fact was present
    in the assembled context. Using two different matchers for those would
    manufacture phantom contamination the moment they disagreed.
    """
    if not normalize(gold):
        return False
    return gold_pattern(gold).search(normalize(haystack)) is not None


def extract_answer(text: str) -> str | None:
    """The span after the last Answer: line, or None if there is no such line."""
    found = _ANSWER_LINE.findall(text)
    return found[-1].strip() if found else None


def _shape(gold: str) -> re.Pattern[str] | None:
    """A pattern matching other values that look like this gold answer.

    Used to catch enumeration. If the answer names two things of the same
    shape, the model listed candidates rather than committing to one.
    """
    g = gold.strip()
    if re.fullmatch(r"[A-Z]{1,4}-\d+", g):
        prefix = g.split("-")[0]
        return re.compile(rf"\b{prefix}-\d+\b", re.IGNORECASE)
    if re.fullmatch(r"\d{2}:\d{2}", g):
        return re.compile(r"\b\d{1,2}:\d{2}\b")
    if re.fullmatch(r"\d+", g):
        return re.compile(r"\b\d[\d,]*\b")
    return None


def classify(
    text: str,
    finish_reason: str,
    fact: Fact,
    *,
    error: str = "",
) -> tuple[Status, str]:
    """Assign exactly one status, in precedence order.

    Service outcomes come first, so a rate limit or a truncation can never be
    read as the model getting an answer wrong.
    """
    if finish_reason in ("oversized", "rate_limited", "api_error"):
        return finish_reason, ""  # type: ignore[return-value]
    if error:
        return "api_error", ""
    if finish_reason == "length":
        # A truncated response that still carried a well-formed answer line is
        # a usable answer; only a truncation that lost the answer is a failure.
        if extract_answer(text) is None:
            return "truncated", ""

    answer = extract_answer(text)
    if answer is None:
        return "unparseable", text.strip()[:120]

    if normalize(answer) == normalize(UNKNOWN_TOKEN) or UNKNOWN_TOKEN.lower() in answer.lower():
        # For a negative probe, declining IS the correct answer.
        return ("correct" if fact.zone == "negative" else "abstained"), answer

    if fact.zone == "negative":
        return "wrong", answer  # named a value for something never planted

    shape = _shape(fact.value)
    if shape is not None and len({m.lower() for m in shape.findall(answer)}) > 1:
        return "unparseable", answer

    return ("correct" if matches(fact.value, answer) else "wrong"), answer


if __name__ == "__main__":
    def f(value: str, zone: str = "deep") -> Fact:
        return Fact(id="t", turn=1, zone=zone, carrier="prose", value=value, probe="?")

    # The substring bug this grader exists to avoid.
    assert not matches("42", "The answer is 142")
    assert not matches("no", "Now, Sunil is taller")
    assert matches("42", "Answer: 42")
    assert matches("INC-4471", "answer: inc 4471"), "separator differences are formatting"

    # Formatting normalisation.
    assert matches("M-88213", "M88213")
    assert matches("18420355", "18,420,355")
    assert matches("1,84,203.55", "₹184203.55")
    assert matches("21:14", "21:14 IST")

    # Typographic traps.
    assert matches("Nandita Rao", "it was Nandita Rao")
    assert normalize("you’re") == "you're"

    # Answer-span extraction, not the whole completion.
    reasoning = "It could be INC-4471 or INC-4468.\nAnswer: INC-4471"
    assert classify(reasoning, "stop", f("INC-4471"))[0] == "correct"
    enumerated = "Answer: possibly INC-4471, INC-4472, or INC-4468"
    assert classify(enumerated, "stop", f("INC-4471"))[0] == "unparseable", (
        "naming several candidates of the gold's shape is not an answer"
    )
    assert classify("I think it was INC-4471", "stop", f("INC-4471"))[0] == "unparseable"

    # The status ladder.
    assert classify("Answer: UNKNOWN", "stop", f("4712"))[0] == "abstained"
    assert classify("Answer: UNKNOWN", "stop", f("UNKNOWN", "negative"))[0] == "correct"
    assert classify("Answer: 47", "stop", f("UNKNOWN", "negative"))[0] == "wrong"
    assert classify("", "oversized", f("4712"))[0] == "oversized"
    assert classify("", "rate_limited", f("4712"))[0] == "rate_limited"
    assert classify("thinking about it", "length", f("4712"))[0] == "truncated"
    assert classify("blah\nAnswer: 4712", "length", f("4712"))[0] == "correct", (
        "a truncation that still landed the answer line is a usable answer"
    )
    assert classify("Answer: 9999", "stop", f("4712"))[0] == "wrong"

    for s in ("truncated", "oversized", "rate_limited", "api_error", "unparseable"):
        assert s in NON_SCORED

    print("grader: substring, formatting, typographic, span and status ladder all hold")
