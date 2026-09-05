"""What counts as a right answer, per probe category.

Inputs:  a probe and the model's reply
Outputs: a dict of named checks, each True or False

Every category returns several named booleans rather than one verdict, because
a single number hides which half of the behaviour an arm got right. "Did not
assert the stale value" and "said it might be out of date" are different
achievements, and an arm can manage the first without the second.

The aged and scheduled categories have no gold value on purpose. There is no
correct current answer for a fact whose shelf life has passed - the correct
behaviour is to say so, which is why asserting the old value flatly is the
failure and hedging it is not.
"""

from __future__ import annotations

from src import matcher
from src.fixture import Probe


def score(probe: Probe, reply: str) -> dict[str, bool]:
    """Grade one reply against one probe."""
    if probe.category in {"announced", "source_drift"}:
        return {
            "states current value": matcher.states_current(reply, probe.gold),
            "does not assert the old one": not matcher.asserts_stale(reply, probe.stale),
        }
    if probe.category == "scope_pair":
        return {
            "states its own value": matcher.states_current(reply, probe.gold),
            "does not state the other context's": not matcher.matches(
                probe.stale, matcher.extract_answer(reply)),
            "does not abstain": not matcher.abstained(reply),
        }
    if probe.category == "aged":
        # Abstaining is safe but useless, and an arm with no memory at all does
        # it for free - so "did not assert a stale fact" on its own measures
        # nothing. The behaviour worth having is BOTH halves: still hand over
        # the last thing you knew, and say plainly that it may have moved on.
        return {
            "does not assert a stale fact": not matcher.asserts_stale(reply, probe.stale),
            "still offers the last known value": matcher.matches(probe.stale, reply),
            "says it is unverified": matcher.hedged(reply),
        }
    if probe.category == "scheduled":
        return {
            "does not present it as current": not matcher.asserts_stale(reply, probe.stale),
            "flags that it ended, or abstains": matcher.hedged(reply) or matcher.abstained(reply),
        }
    if probe.category == "stable_control":
        return {
            "states the value": matcher.states_current(reply, probe.gold),
            "does not hedge a stable fact": not matcher.hedged(reply),
        }
    if probe.category == "in_message":
        return {"states the value": matcher.matches(probe.gold, reply)}
    if probe.category == "negative":
        return {"abstains": matcher.abstained(reply)}
    raise ValueError(f"unknown category {probe.category!r}")


def passed(checks: dict[str, bool]) -> bool:
    """A probe passes only when every one of its checks passes."""
    return bool(checks) and all(checks.values())


HEADLINE_CATEGORIES = ("source_drift",)
"""The categories the criterion is stated over.

Only source drift. The aged probes are handled almost as well by simply showing
a fact's age at read time, so crediting the sweep for them would overstate it.
Re-reading a source nobody mentioned is the thing only the sweep does.
"""


if __name__ == "__main__":
    drift = Probe("b-x", "source_drift", "What is my desk?", gold="Desk 9A-02", stale="Desk 4C-17")
    assert passed(score(drift, "Answer: Desk 9A-02"))
    assert not passed(score(drift, "Answer: Desk 4C-17")), "the stale value must fail"
    assert not passed(score(drift, "Answer: UNKNOWN"))

    aged = Probe("c-x", "aged", "What is my sprint ticket right now?", stale="PAY-33187")
    assert not passed(score(aged, "Answer: PAY-33187")), "asserting a stale fact is the failure"
    assert passed(score(aged, "Answer: PAY-33187 (unverified since 2026-06-01)")), \
        "the same value, flagged, is the right answer"
    assert not passed(score(aged, "Answer: UNKNOWN")), \
        "abstaining is safe but useless, and an arm with no memory gets it for free"

    stable = Probe("d-x", "stable_control", "Where did I grow up?", gold="Thrissur")
    assert passed(score(stable, "Answer: Thrissur"))
    assert not passed(score(stable, "Answer: Thrissur (unverified since 2025-01-01)")), \
        "a stable fact hedged is a false positive, and it has to cost something"

    pair = Probe("s-x", "scope_pair", "What do I want for work notifications?",
                 gold="Slack only", stale="WhatsApp only")
    assert passed(score(pair, "Answer: Slack only"))
    assert not passed(score(pair, "Answer: WhatsApp only"))

    negative = Probe("n-x", "negative", "What is my shoe size?")
    assert passed(score(negative, "Answer: UNKNOWN"))
    assert not passed(score(negative, "Answer: 9"))
    print("OK - each category rewards the behaviour it is meant to measure")
