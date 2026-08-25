"""The escalation gate: decide whether the small model's answer can be trusted.

Inputs:  an item, the SLM's answer, and a strictness level.
Outputs: (escalate?, which check fired)

This module is the entire project in miniature. A cascade only saves money if
the thing deciding whether to escalate is cheaper than the model it escalates
to - so every check here runs locally and costs nothing per token. The moment
you ask the big model "was that answer good?", you have paid the big model's
price on every single query and saved exactly nothing.

The checks are ordered cheapest-first and each level is a superset of the one
below, so escalation rate rises monotonically with strictness. That monotonicity
is what lets the strictness grid trace a clean deferral curve.

None of these checks knows the right answer. They only ask "does this look like
a confident, well-formed reply?" - a verifier that could tell right from wrong
would make the small model unnecessary.
"""

from __future__ import annotations

from src.grader import extract_answer, format_ok, normalize

# Phrases a model reaches for when it is guessing. Cheap to detect and
# surprisingly informative, because a 4B model that hedges is usually correct
# to hedge.
HEDGE_MARKERS = (
    "i'm not sure",
    "i am not sure",
    "not entirely sure",
    "i don't know",
    "i do not know",
    "cannot determine",
    "can't determine",
    "unable to determine",
    "insufficient information",
    "not enough information",
    "it depends",
    "as an ai",
    "i cannot",
    "unclear",
)

NEVER = 0
EMPTY = 1
FORMAT = 2
HEDGE = 3
DISAGREE = 4
ALWAYS = 5


def is_empty(raw_answer: str) -> bool:
    """No extractable answer at all."""
    return not extract_answer(raw_answer).strip()


def is_hedging(raw_answer: str) -> bool:
    """The reply contains an explicit low-confidence marker."""
    text = normalize(raw_answer)
    return any(marker in text for marker in HEDGE_MARKERS)


def disagrees(samples: list[str]) -> bool:
    """Independent re-samples of the same question disagree with each other.

    Self-consistency is the strongest local signal available. When a model is
    on solid ground it lands on the same answer every time; when it is guessing,
    the guesses scatter. It costs k extra local calls - real wall-clock, zero
    dollars - which is a trade the ledger makes visible rather than hides.
    """
    normalized = {normalize(extract_answer(sample)) for sample in samples}
    normalized.discard("")
    return len(normalized) > 1


def should_escalate(
    item: dict,
    raw_answer: str,
    samples: list[str],
    strictness: int,
) -> tuple[bool, str]:
    """Return (escalate?, reason) at the given strictness level."""
    if strictness <= NEVER:
        return False, ""
    if strictness >= ALWAYS:
        return True, "always"

    if is_empty(raw_answer):
        return True, "empty"
    if strictness < FORMAT:
        return False, ""

    if not format_ok(raw_answer, item.get("choices"), item.get("numeric", False)):
        return True, "format"
    if strictness < HEDGE:
        return False, ""

    if is_hedging(raw_answer):
        return True, "hedge"
    if strictness < DISAGREE:
        return False, ""

    if samples and disagrees(samples):
        return True, "disagree"

    return False, ""


if __name__ == "__main__":
    numeric_item = {"numeric": True, "choices": None}
    choice_item = {"numeric": False, "choices": ["billing", "technical"]}

    cases = [
        ("empty output escalates at 1", numeric_item, "", [], 1, True),
        ("good numeric answer stays at 4", numeric_item, "Answer: 42", ["Answer: 42"] * 3, 4, False),
        ("non-numeric answer to a numeric item", numeric_item, "Answer: about forty", [], 2, True),
        ("valid label passes format", choice_item, "Answer: billing", ["Answer: billing"] * 3, 2, False),
        ("invalid label fails format", choice_item, "Answer: refunds", [], 2, True),
        ("hedging escalates at 3", numeric_item, "Answer: 42, but I'm not sure", [], 3, True),
        ("hedging ignored at 2", numeric_item, "Answer: 42, but I'm not sure", [], 2, False),
        ("disagreement escalates at 4", numeric_item, "Answer: 42",
         ["Answer: 42", "Answer: 51", "Answer: 42"], 4, True),
        ("agreement passes at 4", numeric_item, "Answer: 42",
         ["Answer: 42", "Answer: 42", "Answer: 42"], 4, False),
        ("strictness 0 never escalates", numeric_item, "", [], 0, False),
        ("strictness 5 always escalates", numeric_item, "Answer: 42", ["Answer: 42"] * 3, 5, True),
    ]

    failures = 0
    for label, item, answer, samples, level, expected in cases:
        got, reason = should_escalate(item, answer, samples, level)
        ok = got == expected
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} L{level} {label:44s} -> {got} ({reason or '-'})")

    print()
    if failures:
        raise SystemExit(f"{failures} verifier test(s) failed")
    print(f"all {len(cases)} verifier tests passed")
