"""Grading answers without an LLM judge, and without flattering long ones.

Inputs:  a model's reply and a gold value
Outputs: booleans - did it say the value, did it hedge, did it abstain

Substring matching is the trap this file exists to avoid. `expected in answer`
scores "42" as present inside "142", and "no" as present inside "Now, ...". It
systematically rewards verbose hedging answers, which is precisely the failure
mode of an arm that has no memory and is padding. Everything here is anchored
on word boundaries built from the gold value.

The same matcher grades the run and checks the fixture's own independence
gates. A gate that disagreed with the grader would report leaks the scorecard
cannot see, and miss ones it can.
"""

from __future__ import annotations

import re
import unicodedata

# Models emit typographic punctuation, and an ASCII apostrophe in a pattern
# then silently never matches - a false negative that looks like a memory miss.
_SMART = {
    0x2018: "'", 0x2019: "'", 0x201C: '"', 0x201D: '"',
    0x2013: "-", 0x2014: "-", 0x2212: "-", 0x00A0: " ", 0x202F: " ", 0x2009: " ",
}

ANSWER_PREFIX = "Answer:"
UNKNOWN = "UNKNOWN"
HEDGE_RE = re.compile(
    r"\b(unverified|unconfirmed|not\s+confirmed|may\s+be\s+out\s+of\s+date|"
    r"might\s+have\s+changed|no\s+longer\s+current|needs?\s+verification|"
    r"has\s+ended|expired|last\s+confirmed|as\s+of\b|please\s+confirm|"
    r"i'?m\s+not\s+sure|out\s+of\s+date)",
    re.IGNORECASE,
)


def normalize(text: str) -> str:
    """Fold typography, money formatting and whitespace.

    Digit-group commas and the rupee sign come off because a model writes the
    same amount as "Rs 41,904", "41904" and "Rs41,904" from one answer to the
    next, and none of those differences is a wrong answer. The comma is dropped
    only between digits, so "Pune, Maharashtra" keeps its comma and does not
    silently become one word.
    """
    folded = unicodedata.normalize("NFKC", text or "").translate(_SMART)
    folded = folded.replace("\u20b9", " ")
    folded = re.sub(r"(?<=\d),(?=\d)", "", folded)
    return re.sub(r"\s+", " ", folded).strip()


def extract_answer(reply: str) -> str:
    """Pull the value off the anchored final line, or fall back to the last line."""
    text = normalize(reply)
    matches = re.findall(rf"{ANSWER_PREFIX}\s*(.+)", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].strip().rstrip(".")
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[-1].strip() if lines else ""


def gold_pattern(gold: str) -> re.Pattern[str]:
    """A boundary-anchored pattern for one gold value.

    Separators are handled inside the pattern rather than by stripping them from
    the text: stripping hyphens turns INC-4471 into inc4471, and then no
    boundary survives to stop xinc4471y matching.
    """
    parts = [re.escape(part) for part in re.split(r"[\s\-_/]+", normalize(gold)) if part]
    body = r"[\s\-_/]*".join(parts) if parts else re.escape(normalize(gold))
    lead = r"(?<![0-9A-Za-z])" if body else ""
    trail = r"(?![0-9A-Za-z])"
    return re.compile(lead + body + trail, re.IGNORECASE)


def matches(gold: str, text: str) -> bool:
    """Does `text` actually state `gold`?"""
    if not gold:
        return False
    return bool(gold_pattern(gold).search(normalize(text)))


def count_occurrences(gold: str, text: str) -> int:
    """How many times `gold` appears - used by the fixture's uniqueness gates."""
    return len(gold_pattern(gold).findall(normalize(text)))


def abstained(reply: str) -> bool:
    """Did the model decline to answer?"""
    answer = extract_answer(reply).upper()
    return UNKNOWN in answer or not answer


def hedged(reply: str) -> bool:
    """Did the model mark its answer as possibly out of date?"""
    return bool(HEDGE_RE.search(normalize(reply)))


def states_current(reply: str, gold: str) -> bool:
    """States the right value, and does not disown it in the same breath."""
    return matches(gold, extract_answer(reply)) and not abstained(reply)


def asserts_stale(reply: str, stale_gold: str) -> bool:
    """States an out-of-date value AS FACT - the failure this project is about.

    Saying the old value while flagging it is a different, much better outcome
    than saying it flatly, so the hedge is what separates them.
    """
    return matches(stale_gold, extract_answer(reply)) and not hedged(reply) and not abstained(reply)


if __name__ == "__main__":
    # The bug this file exists to prevent, taken from a real harness in this repo.
    assert not matches("42", "The answer is 142"), "substring matching inflates every arm"
    assert not matches("no", "Now, Sunil is taller")
    assert not matches("50", "500 requests per minute")
    assert matches("42", "Answer: 42.")
    assert matches("50", "Answer: 50 requests")

    # Separators live in the pattern, so every spelling of one id matches and
    # nothing longer does.
    for spelling in ("INC-4471", "INC 4471", "inc4471", "inc_4471"):
        assert matches("INC-4471", f"Answer: {spelling}"), spelling
    assert not matches("INC-4471", "Answer: xINC-4471y")

    assert extract_answer("blah\nAnswer: Bengaluru.") == "Bengaluru"
    assert extract_answer("Answer: UNKNOWN") == "UNKNOWN"
    assert abstained("Answer: UNKNOWN")
    assert not abstained("Answer: Pune")

    # Typographic punctuation must not break a match.
    assert matches("Cult Fit", "Answer: Cult Fit")
    assert hedged("Answer: Honda City (unverified since 2026-03-01)")
    assert not hedged("Answer: Honda City")

    # Stating a stale value flatly is the failure; flagging it is not.
    assert asserts_stale("Answer: Pune", "Pune")
    assert not asserts_stale("Answer: Pune (unverified since 2026-02-21)", "Pune")
    assert not asserts_stale("Answer: UNKNOWN", "Pune")
    assert states_current("Answer: Bengaluru", "Bengaluru")

    # Indian money formatting: one amount, many spellings, all correct.
    for spelling in ("Rs 41,904", "41,904", "41904", "\u20b941,904", "Rs. 41904"):
        assert matches("41,904", f"Answer: {spelling}"), spelling
    # ... and a longer amount that merely contains those digits is not.
    assert not matches("41,904", "Answer: Rs 5,41,904")
    assert not matches("1904", "Answer: Rs 41,904")
    # Percentages and clock times keep their bare form.
    assert matches("2.375", "Answer: 2.375%")
    assert matches("14:37", "Answer: 14:37 IST")
    assert not matches("14:37", "Answer: 114:37")
    print("OK - boundary matching, money formats, separators, typography, hedges, stale")
