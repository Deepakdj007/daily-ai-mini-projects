"""Deterministic exact-match grading.

Inputs:  raw model output plus an item's gold answer.
Outputs: a boolean verdict, with no model in the loop.

Why there is no LLM judge here
------------------------------
An LLM judge would have to grade answers from the same model family it is
judging, which is a documented source of self-preference bias. Worse, in this
project the judge's verdicts would decide which tier "wins", so the bias would
land directly on the headline. Authoring every item to have a short, checkable
gold answer removes the judge entirely.

The bug this module exists to avoid
-----------------------------------
The obvious implementation is `gold in answer` after normalisation, which is
what agent-eval-arena/src/evaluators.py does today. It is wrong:

    gold 'no' vs 'Now, Sunil is taller'  -> True   ("no" inside "Now")
    gold 'no' vs "I don't know"          -> True   ("no" inside "know")
    gold '42' vs 'The answer is 142'     -> True

Substring containment systematically rewards long hedging answers, which is
precisely the small model's failure mode - so the bug would inflate the SLM
tier and corrupt the entire comparison.
"""

from __future__ import annotations

import re
import unicodedata

from src.config import ANSWER_PREFIX

_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_WS_RE = re.compile(r"\s+")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_ORPHAN_THINK_RE = re.compile(r"^.*?</think>", re.DOTALL | re.IGNORECASE)

# Typographic punctuation folded to its ASCII equivalent, keyed by code point.
# Written as ordinals rather than literal characters on purpose: a curly quote
# copied through a PDF, a terminal or an editor can silently arrive as a
# straight one, which turns the replacement into a no-op that fails quietly.
_PUNCT_FOLD = {
    0x2018: "'",  # left single quotation mark
    0x2019: "'",  # right single quotation mark / apostrophe
    0x201C: '"',  # left double quotation mark
    0x201D: '"',  # right double quotation mark
    0x2013: "-",  # en dash
    0x2014: "-",  # em dash
}


def strip_thinking(text: str) -> str:
    """Remove reasoning blocks that leaked into the visible answer.

    qwen3.5 is a thinking model. We disable thinking via the API, but a leaked
    <think> block is not merely verbose - it would be graded as part of the
    answer and would tank the small tier's score for a reason that has nothing
    to do with its ability. Handles the unclosed case too, which happens when
    generation hits the token cap mid-thought.
    """
    text = _THINK_RE.sub(" ", text)
    if "</think>" in text.lower():
        text = _ORPHAN_THINK_RE.sub(" ", text)
    return text.strip()


def extract_answer(text: str) -> str:
    """Pull the value off the final 'Answer: X' line.

    Anchoring on an explicit marker is what makes word-boundary matching safe:
    it shrinks the haystack from a whole paragraph to a single short span, so a
    stray word in the model's reasoning cannot satisfy the match.
    """
    text = strip_thinking(text)
    if not text:
        return ""

    marker = ANSWER_PREFIX.lower().rstrip(":")
    best = ""
    for line in text.splitlines():
        stripped = line.strip().lstrip("*# ").strip()
        low = stripped.lower()
        if low.startswith(marker):
            remainder = stripped[len(marker):].lstrip(": \t")
            if remainder:
                best = remainder
    if best:
        return best.strip()

    # No marker: fall back to the last non-empty line, which is where a short
    # answer almost always lands.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def normalize(text: str) -> str:
    """Lowercase, fold unicode punctuation, drop noise, collapse whitespace.

    Curly quotes and narrow no-break spaces show up constantly in model output
    and would otherwise cause false negatives against plainly-typed golds.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_PUNCT_FOLD)
    text = text.lower().strip()
    text = text.strip("`*_ ")
    text = re.sub(r"[.,;:!?]+$", "", text)
    return _WS_RE.sub(" ", text).strip()


def _as_number(text: str) -> float | None:
    """Parse a bare numeric value, tolerating thousands separators and $/%."""
    cleaned = text.strip().strip("$%").replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _numbers_in(text: str) -> list[float]:
    """Every number appearing in a span of text, commas removed."""
    out: list[float] = []
    for raw in _NUMBER_RE.findall(text):
        value = _as_number(raw)
        if value is not None:
            out.append(value)
    return out


def grade(raw_output: str, gold: str) -> bool:
    """True when the model's answer matches the gold answer.

    Numeric golds compare numerically, so '1,234' == '1234' == '1234.0' and
    '142' never satisfies a gold of '42'. Text golds compare on whole words
    against the extracted answer span only.
    """
    answer = extract_answer(raw_output)
    if not answer:
        return False

    gold_norm = normalize(gold)
    answer_norm = normalize(answer)
    if not gold_norm:
        return False

    gold_number = _as_number(gold_norm)
    if gold_number is not None:
        candidates = _numbers_in(answer_norm)
        if not candidates:
            return False
        # Accept the answer if it is exactly the gold number. Compare against
        # every number present because models write "Answer: 42 apples".
        return any(abs(candidate - gold_number) < 1e-9 for candidate in candidates)

    if answer_norm == gold_norm:
        return True

    # Whole-word match, never a bare substring. This is the line that fixes the
    # inherited bug.
    pattern = r"(?<!\w)" + re.escape(gold_norm) + r"(?!\w)"
    return re.search(pattern, answer_norm) is not None


def format_ok(raw_output: str, choices: list[str] | None, numeric: bool) -> bool:
    """Cheap structural check: did the answer even take the required shape?

    Used by the verifier as its first and cheapest escalation signal. It asks
    only 'is this well-formed', never 'is this right' - a verifier that could
    tell right from wrong would make the whole project unnecessary.
    """
    answer = normalize(extract_answer(raw_output))
    if not answer:
        return False
    if numeric:
        return bool(_numbers_in(answer))
    if choices:
        return any(
            re.search(r"(?<!\w)" + re.escape(normalize(c)) + r"(?!\w)", answer)
            for c in choices
        )
    return True


if __name__ == "__main__":
    # The four cases that the inherited substring matcher gets wrong. If any of
    # these regress, every number the project reports becomes untrustworthy.
    regressions = [
        ("no", "Answer: Now, Sunil is taller", False),
        ("no", "Answer: I don't know", False),
        ("yes", "Answer: Yesterday it rained", False),
        ("42", "Answer: The answer is 142", False),
    ]
    # Cases that must still pass.
    positives = [
        ("no", "Answer: no", True),
        ("42", "Answer: 42", True),
        ("42", "The result is 42.\nAnswer: 42 apples", True),
        ("1234", "Answer: 1,234", True),
        ("billing", "Answer: Billing", True),
        ("18", "<think>let me add 9+9</think>\nAnswer: 18", True),
        ("18", "<think>unclosed reasoning\nAnswer: 18", True),
        # Typographic apostrophe must fold to the ASCII one.
        ("don't", "Answer: don’t", True),
    ]

    failures = 0
    for gold, output, expected in regressions + positives:
        got = grade(output, gold)
        ok = got == expected
        failures += not ok
        flag = "ok  " if ok else "FAIL"
        print(f"{flag} gold={gold!r:12} out={output!r:48} -> {got} (want {expected})")

    print()
    if failures:
        raise SystemExit(f"{failures} grader test(s) failed")
    print(f"all {len(regressions) + len(positives)} grader tests passed")
