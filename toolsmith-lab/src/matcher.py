"""Deciding whether an answer is the right answer.

Inputs:  a value from the model or from a generated tool, and the gold value
Outputs: a normalised form, and a match verdict

One matcher, used in three places: grading the run, checking a generated tool
against the held-out inputs, and the pre-run gate that re-derives every gold
answer from the oracle. A gate that disagrees with the grader reports failures
the scorecard cannot see and misses ones it can, so there is exactly one
implementation and all three import it.

The modes exist because the families answer in different shapes. A model that
says "Rs 10,82,776.63" and an oracle that says 1082776.63 agree, and a matcher
that cannot see that is measuring formatting.
"""

from __future__ import annotations

import re
import unicodedata

MODES = ("str_ci", "int", "money2", "words", "exact")

_NUMBER = re.compile(r"-?\d[\d,_\s]*(?:\.\d+)?")
# Models reach for typographic punctuation unprompted: a non-breaking space
# inside a number, a right single quote for an apostrophe, an en dash for a
# minus. None of them should decide whether an answer is correct.
_PUNCT = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"',
                        "–": "-", "—": "-", "−": "-"})


def _clean(value: object) -> str:
    """Collapse whitespace and fold the punctuation models vary on."""
    text = unicodedata.normalize("NFKC", str(value)).translate(_PUNCT)
    return " ".join(text.split()).strip()


def _number(value: object) -> float | None:
    """The first number in a value, ignoring currency words and separators."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = _NUMBER.search(_clean(value))
    if match is None:
        return None
    try:
        return float(re.sub(r"[,_\s]", "", match.group()))
    except ValueError:
        return None


def normalise(value: object, mode: str) -> object:
    """The comparable form of a value under one mode."""
    if mode == "int":
        number = _number(value)
        return None if number is None else int(round(number))
    if mode == "money2":
        number = _number(value)
        return None if number is None else round(number + 0.0, 2)
    if mode == "str_ci":
        return _clean(value).upper()
    if mode == "words":
        text = _clean(value).lower()
        text = re.sub(r"\b(rupees|rupee|only|and)\b", " ", text)
        return " ".join(re.sub(r"[^a-z0-9 ]", " ", text).split())
    if mode == "exact":
        return _clean(value)
    raise ValueError(f"unknown compare mode {mode!r}; expected one of {MODES}")


def matches(got: object, want: object, mode: str) -> bool:
    """Whether a value counts as the gold answer under one mode.

    money2 compares within half a paisa rather than exactly, because the same
    schedule summed in a different order lands a hundredth apart and that is
    not a wrong answer.
    """
    left, right = normalise(got, mode), normalise(want, mode)
    if left is None or right is None:
        return False
    if mode == "money2":
        return abs(float(left) - float(right)) < 0.005
    return left == right


def contains_answer(text: str, want: object, mode: str) -> bool:
    """Whether a free-text reply states the gold answer somewhere in it.

    Used only by the gate that checks an answer is not sitting in its own
    prompt. Grading never uses this: a reply long enough will contain almost
    any short answer by accident, which is the substring bug that inflated
    every arm of an earlier project in this series.
    """
    target = normalise(want, mode)
    if target is None:
        return False
    if mode in ("int", "money2"):
        return any(matches(found, want, mode) for found in _NUMBER.findall(_clean(text)))
    return str(target) in normalise(text, mode)


if __name__ == "__main__":
    assert matches("Rs 10,82,776.63", 1082776.63, "money2")
    assert matches(1082776.628, 1082776.63, "money2")
    assert not matches(1082776.70, 1082776.63, "money2")
    assert matches(" v ", "V", "str_ci")
    assert matches("4", 4, "int") and matches(4.0, 4, "int")
    assert not matches("four", 4, "int")
    assert matches("One Crore Five Thousand", "one crore five thousand", "words")
    assert matches("rupees one lakh only", "one lakh", "words")
    assert matches("1.0.0-alpha.10", "1.0.0-alpha.10", "exact")
    assert not matches("1.0.0", "1.0.0-alpha.10", "exact")
    assert not matches("", "V", "str_ci")
    assert not matches(None, 4, "int")
    # Booleans must not read as numbers: True would otherwise match 1.
    assert not matches(True, 1, "int")

    assert contains_answer("the check character is V, so it is valid", "V", "str_ci")
    assert contains_answer("total interest comes to 10,82,776.63", 1082776.63, "money2")
    assert not contains_answer("total interest comes to 9,00,000.00", 1082776.63, "money2")
    try:
        normalise("x", "nope")
        raise SystemExit("an unknown mode must raise")
    except ValueError:
        pass
    print("OK - matcher")
