"""The four probes and their assertions.

Each probe is built so exactly one memory tier can supply the answer. That is a
design constraint, not a hope, and two of these were rewritten during the build
because the first versions did not hold:

- Probe B originally blamed a firewall egress allowlist. "Check your firewall
  rules" is advice the model gives from general knowledge, so the probe passed
  with episodic memory switched off. Its root cause is now a PaySetu-specific
  auto-pause threshold that has to be cleared by hand — not guessable.
- Probe B's customer was originally seeded with PaySetu-hosted webhooks, and the
  agent rightly refused to apply a self-hosting lesson to them. The probe
  customer now satisfies the precondition.

Scoring is programmatic. Nothing here asks the model whether it followed a rule.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Models write typographic punctuation: "you’re" with U+2019, not an ASCII
# apostrophe. Every pattern below runs against normalised text, because a regex
# containing a plain ' silently fails to match and the check reports a false
# negative that looks exactly like a memory failure.
_SMART = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "‑": "-", "–": "-", " ": " ", " ": " "})


def _norm(text: str) -> str:
    """Fold typographic punctuation to ASCII so the patterns can match."""
    return text.translate(_SMART)


# Phrases that mean the agent asked for something it was already told.
_REASK = re.compile(r"\b(which|what)\b[^.?]{0,40}\b(plan|tier|region)\b", re.I)

# Attribution: the limit is stated as THIS customer's, not listed as one row in
# a price table. This is the real signal for semantic memory — an agent without
# it still prints "500", inside "Starter 100, Growth 500, Scale 2000".
_ATTRIBUTES_GROWTH = re.compile(
    r"(your\s+growth\s+plan|you(?:'re|\s+are)\s+on\s+the\s+growth\s+plan)", re.I
)

# The arbitrary auto-pause threshold from the seeded episode. \b50\b rather than
# "50" so it cannot be satisfied by the 500 req/min figure.
_PAUSE_THRESHOLD = re.compile(r"\b50\b")
_RESUME_ACTION = re.compile(r"(resum|re-?enabl|reactivat)", re.I)

# An opening apology or throat-clearing.
_APOLOGY_OPENER = re.compile(
    r"^\W*(hi|hello|dear|thanks|thank you|sorry|we('|’)?re sorry|apolog|unfortunately|"
    r"we understand|we regret)",
    re.I,
)

# A promised resolution time. Stating the documented T+1 window is fine; telling
# the customer when their money will actually land is not.
_TIMELINE_PROMISE = re.compile(
    r"(next business day|within \d+\s*(hour|day|business)|by (tomorrow|today|end of)|"
    r"expected (arrival|resolution|time)|will (arrive|be credited|be resolved) (by|within|on)|"
    r"should (arrive|land) (by|within))",
    re.I,
)


@dataclass
class Probe:
    """One ticket plus the checks that decide whether memory did its job."""

    key: str
    measures: str  # which tier this probe is sensitive to
    customer_id: str
    ticket: str
    # Tokens that must be absent from the assembled context when the measured
    # tier is off. If they show up, some other tier is leaking the answer and the
    # probe is broken rather than passing.
    answer_tokens: tuple[str, ...]

    def score(self, state: dict[str, Any]) -> dict[str, bool]:
        """Run this probe's checks against a finished turn."""
        return SCORERS[self.key](state)


def _lines(reply: str) -> list[str]:
    return [ln for ln in reply.strip().splitlines() if ln.strip()]


def _score_a(state: dict[str, Any]) -> dict[str, bool]:
    """Semantic: does it attribute the limit to THIS customer?

    The discriminator is attribution, not the number. Without semantic memory the
    model still prints "500" — inside "Starter 100, Growth 500, Scale 2000",
    because it does not know which row applies. A bare `"500" in reply` check
    therefore passes with the tier switched off and measures nothing.

    "Mentions no other tier" was the next attempt and was also wrong: it failed
    whenever the playbook was off, because a verbose answer lists every tier for
    completeness even though it knows the customer's own. That made probe A
    sensitive to procedural memory, which is exactly the cross-tier bleed the
    probes are supposed to avoid.

    An attribution regex ("your Growth plan") was the third attempt and was still
    verbosity-sensitive: a long answer phrases it as "Growth tier" or "your
    account is on Growth", so the check failed whenever the playbook was off.

    What finally holds is the absence of the price table. Knowing which plan the
    customer is on means not having to list all three. Naming Scale as an upgrade
    is fine; reciting Starter AND Scale is the tell that the agent is guessing.
    """
    reply = _norm(state["reply"])
    low = reply.lower()
    return {
        "names Growth": "growth" in low,
        "does not recite the price table": not ("starter" in low and "scale" in low),
        "does not re-ask the plan": not _REASK.search(reply),
    }


def _score_b(state: dict[str, Any]) -> dict[str, bool]:
    """Episodic: does it recall the specific auto-pause mechanism?

    Checks the arbitrary threshold, not the concept. "The subscription may be
    paused" and "deliveries will resume" are things the model says from general
    knowledge, so `paus`/`resum` alone passed with episodic memory off on some
    runs. "50 consecutive failures" is a number that exists nowhere except the
    seeded episode — either it was read or it was not.
    """
    reply = _norm(state["reply"])
    return {
        "recalls the 50-failure threshold": bool(_PAUSE_THRESHOLD.search(reply)),
        "names the resume action": bool(_RESUME_ACTION.search(reply)),
    }


def _score_c(state: dict[str, Any]) -> dict[str, bool]:
    """Procedural: four conduct rules, reported separately."""
    reply = _norm(state["reply"])
    return {
        "escalates (>24h rule)": state["escalate"] is True,
        "5 lines or fewer": len(_lines(reply)) <= 5,
        "no apology opener": not _APOLOGY_OPENER.match(reply),
        "promises no timeline": not _TIMELINE_PROMISE.search(reply),
    }


def _score_control(state: dict[str, Any]) -> dict[str, bool]:
    """Control: answerable from the static docs alone, so every config passes.

    Checks the two doc facts the question actually asks for, and stays away from
    exact phrasing. An earlier version required the literal words "dashboard" or
    "developer" and failed one run purely on wording — a flaky control probe is
    worse than no control probe, because it discredits the whole scorecard.
    """
    reply = _norm(state["reply"])
    return {
        "explains rotation": "rotat" in reply.lower(),
        "gives the 24h overlap": "24" in reply,
    }


SCORERS = {
    "A": _score_a,
    "B": _score_b,
    "C": _score_c,
    "control": _score_control,
}

PROBES: list[Probe] = [
    Probe(
        key="A",
        measures="semantic",
        customer_id="acme",
        ticket=(
            "We're getting 429s on the payouts endpoint since this morning. "
            "What is our rate limit and why are we hitting it?"
        ),
        answer_tokens=("growth", "500"),
    ),
    Probe(
        key="B",
        measures="episodic",
        # A different customer from the one whose ticket produced the lesson.
        # Episodic memory is global, so a hit here is cross-customer transfer.
        customer_id="beta",
        ticket=(
            "Our webhooks suddenly stopped arriving yesterday and nothing changed "
            "on our side. Here's a request ID from one that never landed: req_5512aa."
        ),
        answer_tokens=("pause", "resum"),
    ),
    Probe(
        key="C",
        measures="procedural",
        customer_id="acme",
        ticket=(
            "Our settlement is 36 hours late and our vendors are chasing us. "
            "When exactly will the money arrive?"
        ),
        answer_tokens=(),  # conduct, not content — nothing to leak
    ),
    Probe(
        key="control",
        measures="none",
        customer_id="beta",
        ticket="How do I rotate my API key, and does the old one stop working straight away?",
        answer_tokens=(),
    ),
]


def context_leak(state: dict[str, Any], probe: Probe) -> list[str]:
    """Check whether a disabled tier's answer is present in the context anyway.

    Args:
        state: the finished turn, whose trace holds everything injected.
        probe: the probe that was run.

    Returns:
        Any answer token found in the assembled context. Non-empty means the
        probe could be answered without the tier it is supposed to measure.
    """
    trace = state.get("trace") or {}
    injected = " ".join(
        [*trace.get("facts", []), *trace.get("episodes", []), *trace.get("rules", [])]
    ).lower()
    return [tok for tok in probe.answer_tokens if tok in injected]


if __name__ == "__main__":
    print(f"{len(PROBES)} probes\n")
    for probe in PROBES:
        headline = (SCORERS[probe.key].__doc__ or "").strip().splitlines()[0]
        print(f"  {probe.key:8} measures {probe.measures:11} customer={probe.customer_id}")
        print(f"           {probe.ticket[:70]}...")
        print(f"           {headline}\n")

    # The patterns are load-bearing, so they get their own checks — including the
    # typographic-apostrophe case, which is what made probe A misreport.
    assert _APOLOGY_OPENER.match("We're sorry you're experiencing a delay.")
    assert _APOLOGY_OPENER.match("Hi Acme Retail,")
    assert not _APOLOGY_OPENER.match("Provide the settlement request ID so we can investigate.")
    assert _TIMELINE_PROMISE.search("funds are transferred by the next business day")
    assert _TIMELINE_PROMISE.search("it will arrive within 24 hours")
    assert not _TIMELINE_PROMISE.search("On the Growth plan settlement is T+1.")
    assert _REASK.search("Which plan are you on?")
    assert not _REASK.search("Your Growth plan permits 500 requests per minute.")

    # Kept because it documents the U+2019 trap that made probe A misreport for
    # two rounds, even though the final probe A check no longer needs it.
    assert _ATTRIBUTES_GROWTH.search("Your Growth plan allows 500 requests per minute.")
    curly = "As you’re on the Growth plan, the limit is 500 requests per minute."
    assert not _ATTRIBUTES_GROWTH.search(curly), "raw curly quote should NOT match"
    assert _ATTRIBUTES_GROWTH.search(_norm(curly)), "normalised curly quote should match"

    knows = _score_a({"reply": "Your Growth plan allows 500 requests per minute."})
    guesses = _score_a({"reply": "Limits are Starter 100, Growth 500, Scale 2000 per minute."})
    upgrade = _score_a({"reply": "You are on Growth (500/min). Consider Scale if you need more."})
    assert all(knows.values()), knows
    assert not all(guesses.values()), guesses
    assert all(upgrade.values()), upgrade

    assert _PAUSE_THRESHOLD.search("auto-paused after 50 consecutive delivery failures")
    assert not _PAUSE_THRESHOLD.search("your plan allows 500 requests per minute")
    assert _RESUME_ACTION.search("click Resume in Dashboard > Webhooks")
    print("pattern checks OK — including the U+2019 apostrophe case")
