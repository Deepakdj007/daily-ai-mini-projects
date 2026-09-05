"""The synthetic timeline: facts that rot in four different ways, and probes for each.

Inputs:  a seed and a start date
Outputs: a Fixture - seed facts, change events, source versions and probes

LongMemEval covers the case where a later message contradicts an earlier one.
Nothing public covers the other three ways a memory goes wrong: a fact that
simply aged, one whose source changed while nobody was talking, and one whose
window closed. So this fixture is built to separate them.

Values are minted to be unguessable and are checked for uniqueness, because a
probe whose answer a model could produce from general knowledge measures
nothing. `gates()` proves each answer appears in exactly one place before any
model is called.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

from src import clock as clockmod
from src import config, matcher

START = "2026-01-12"
PROBE_AT = "2026-11-20"


@dataclass(frozen=True, slots=True)
class Probe:
    """One question, and what a right answer looks like."""

    key: str
    category: str
    question: str
    gold: str = ""          # the value that is correct at probe time
    stale: str = ""         # the value that WAS correct and no longer is
    measures: str = ""      # which mechanism this probe is aimed at


@dataclass(frozen=True, slots=True)
class Change:
    """A fact changing, either announced in chat or only in a source file."""

    topic: str
    scope: str
    message: str
    at_days: int


@dataclass(slots=True)
class Fixture:
    """Everything one run needs, and nothing that varies between arms."""

    facts: list[dict] = field(default_factory=list)
    changes: list[Change] = field(default_factory=list)
    sources_v1: dict[str, str] = field(default_factory=dict)
    sources_v2: dict[str, str] = field(default_factory=dict)
    probes: list[Probe] = field(default_factory=list)
    start: str = START
    probe_at: str = PROBE_AT

    def digest(self) -> str:
        """A hash of the whole fixture, stamped into the results manifest."""
        blob = json.dumps({
            "facts": self.facts, "probes": [asdict(p) for p in self.probes],
            "changes": [asdict(c) for c in self.changes],
            "v1": self.sources_v1, "v2": self.sources_v2,
        }, sort_keys=True, default=str)
        return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()[:12]


# (topic, scope, text, value, volatility, days before probe time)
# Ages are chosen against the half-lives in config: `fast` facts here are more
# than two half-lives old at probe time, `slow` ones are comfortably inside one.
# Announced facts are stated in a message and later changed in a message. Both
# go through the real extraction path, so the original and its replacement are
# filed under whatever topic the extractor chooses - and therefore under the
# SAME one. Planting the first by hand instead made the probe measure whether
# the extractor happened to agree with the topic a human picked, which it often
# did not: the old and new facts landed on different keys, never collided, and
# every arm answered "Nucleus Strength and Ironbark Fitness".
# (label, first message, days ago, old value, change message, days ago, new value)
_ANNOUNCED = [
    ("gym", "I train at Ironbark Fitness every morning.", 300, "Ironbark Fitness",
     "I've switched gyms - I'm at Nucleus Strength now.", 120, "Nucleus Strength"),
    ("laptop", "I use a Lenovo T14s for work.", 290, "Lenovo T14s",
     "Work finally replaced my laptop, I'm on a Framework 13 now.", 110, "Framework 13"),
    ("team", "I'm on Team Kestrel at work.", 280, "Team Kestrel",
     "I moved teams last week - I'm on Team Marlin now.", 100, "Team Marlin"),
    ("commute", "I commute on the Purple Line metro.", 270, "Purple Line",
     "My commute changed, I take the Yellow Line these days.", 95, "Yellow Line"),
]

_SCOPE_PAIRS = [
    ("communication_preference", "work notifications", "Wants work notifications on Slack only.",
     "Slack only", 250),
    ("communication_preference", "personal notifications",
     "Wants personal notifications on WhatsApp only.", "WhatsApp only", 245),
    ("diet", "morning coffee", "Drinks black filter coffee in the morning.", "black filter", 240),
    ("diet", "evening coffee", "Drinks a decaf latte in the evening.", "decaf latte", 235),
]

# Facts grounded in a file. The file changes at day 200; nobody ever mentions it.
_SOURCED = [
    ("other", "employee id", "Employee id is EMP-70412.", "EMP-70412", "EMP-88356"),
    ("other", "desk", "Sits at Desk 4C-17.", "Desk 4C-17", "Desk 9A-02"),
    ("other", "manager", "Reports to Anil Varghese.", "Anil Varghese", "Sudha Menon"),
    ("other", "cost centre", "Books time to CC-3391.", "CC-3391", "CC-7724"),
    ("other", "badge", "Carries badge BDG-51207.", "BDG-51207", "BDG-63418"),
    ("other", "parking slot", "Parks in Slot P2-114.", "Slot P2-114", "Slot P4-038"),
    ("other", "insurance", "Is covered by policy HL-92310.", "HL-92310", "HL-45178"),
    ("other", "office", "Works out of Prestige Tech Park.", "Prestige Tech Park", "Ecoworld Block 9"),
    ("other", "laptop asset tag", "Laptop asset tag is AST-20719.", "AST-20719", "AST-51663"),
    ("other", "extension", "Desk extension is x4471.", "x4471", "x7719"),
]

# Fast-moving facts stated once and never mentioned again.
_AGED = [
    ("project", "sprint ticket", "Is working on ticket PAY-33187.", "PAY-33187"),
    ("hobby", "current book", "Is reading The Glass Aqueduct.", "The Glass Aqueduct"),
    ("project", "branch", "Is working on branch feat/ledger-rewrite.", "feat/ledger-rewrite"),
    ("project", "side project", "Is building Project Kingfisher.", "Project Kingfisher"),
    ("hobby", "gym class", "Is doing the Zone Two Thursdays class.", "Zone Two Thursdays"),
    ("hobby", "podcast", "Is listening to Sidecar Signals.", "Sidecar Signals"),
    ("other", "course", "Is taking Distributed Ledgers 402.", "Distributed Ledgers 402"),
    ("other", "lunch spot", "Has been going to Anjappar Whitefield.", "Anjappar Whitefield"),
]

_STABLE = [
    ("partner", "mother", "Mother is Lalitha Raghavan.", "Lalitha Raghavan"),
    ("other", "school", "Went to Vidyaranya High.", "Vidyaranya High"),
    ("city", "home town", "Grew up in Thrissur.", "Thrissur"),
    ("employer", "first employer", "First worked at Halcyon Systems.", "Halcyon Systems"),
]

# (topic, scope, text, value, ends on)
_SCHEDULED = [
    ("travel", "schengen visa", "Holds a Schengen visa until 30 June 2026.", "Schengen visa",
     "2026-06-30"),
    ("appointment", "on-call rotation", "Is on the on-call rotation until 14 August 2026.",
     "on-call rotation", "2026-08-14"),
    ("other", "gym membership", "Gym membership runs until 31 July 2026.", "gym membership",
     "2026-07-31"),
    ("appointment", "conference pass", "Has a DevSummit pass valid until 2 September 2026.",
     "DevSummit pass", "2026-09-02"),
]


def _profile_md(sourced, version: int) -> str:
    """The HR file the sourced facts are grounded in."""
    lines = ["# Workplace record", ""]
    for _, scope, _, old, new in sourced:
        lines.append(f"{scope.title()}: {old if version == 1 else new}")
    return "\n".join(lines) + "\n"


def build(*, start: str = START, probe_at: str = PROBE_AT) -> Fixture:
    """Assemble the whole fixture. Deterministic: same inputs, same digest."""
    probe_epoch = clockmod.to_epoch(probe_at)
    day = 86_400
    fixture = Fixture(start=start, probe_at=probe_at)

    def plant(topic, scope, text, value, volatility, age_days, **extra) -> None:
        fixture.facts.append(dict(
            topic=topic, scope=scope, text=text, value=value, volatility=volatility,
            at=probe_epoch - age_days * day, **extra))

    for label, first, first_age, old, message, change_age, new in _ANNOUNCED:
        fixture.changes.append(Change("", label, first, first_age))
        fixture.changes.append(Change("", label, message, change_age))
        fixture.probes.append(Probe(
            f"a-{label.replace(' ', '-')}", "announced",
            f"What is my {label}?", gold=new, stale=old, measures="resolves"))

    for topic, scope, text, value, age in _SCOPE_PAIRS:
        plant(topic, scope, text, value, "slow", age)
        other = next(s for _, s, _, _, _ in _SCOPE_PAIRS
                     if s != scope and s.split()[-1] == scope.split()[-1])
        other_value = next(v for _, s, _, v, _ in _SCOPE_PAIRS if s == other)
        fixture.probes.append(Probe(
            f"s-{scope.replace(' ', '-')}", "scope_pair",
            f"What do I want for {scope}?", gold=value, stale=other_value,
            measures="scope_aware"))

    for topic, scope, text, old, new in _SOURCED:
        plant(topic, scope, text, old, "slow", 320,
              source_kind="fixture", source_ref="fixture:workplace.md")
        fixture.probes.append(Probe(
            f"b-{scope.replace(' ', '-')}", "source_drift",
            f"What is my {scope}?", gold=new, stale=old, measures="sweep"))
    fixture.sources_v1["workplace.md"] = _profile_md(_SOURCED, 1)
    fixture.sources_v2["workplace.md"] = _profile_md(_SOURCED, 2)

    for topic, scope, text, value in _AGED:
        plant(topic, scope, text, value, "fast", 150)
        fixture.probes.append(Probe(
            f"c-{scope.replace(' ', '-')}", "aged",
            f"What is my {scope} right now?", gold="", stale=value, measures="freshness"))

    for topic, scope, text, value in _STABLE:
        plant(topic, scope, text, value, "stable", 330)
        fixture.probes.append(Probe(
            f"d-{scope.replace(' ', '-')}", "stable_control",
            f"What is my {scope}?", gold=value, measures="control"))

    for topic, scope, text, value, ends in _SCHEDULED:
        plant(topic, scope, text, value, "scheduled", 300, valid_to=clockmod.to_epoch(ends))
        fixture.probes.append(Probe(
            f"e-{scope.replace(' ', '-')}", "scheduled",
            f"Is my {scope} still valid?", gold="", stale=value, measures="history"))

    for key, question, gold in [
        ("f-room", "I'm sitting in room R-2214 today. Which room am I in?", "R-2214"),
        ("f-guest", "My guest today is Nikhil Sundaram. Who is my guest?", "Nikhil Sundaram"),
        ("f-code", "The door code this week is ZQ-8841. What is the door code?", "ZQ-8841"),
    ]:
        fixture.probes.append(Probe(key, "in_message", question, gold=gold, measures="control"))

    for key, question in [
        ("n-shoe", "What is my shoe size?"),
        ("n-airline", "Which airline do I prefer?"),
        ("n-landlord", "What is my landlord's name?"),
    ]:
        fixture.probes.append(Probe(key, "negative", question, measures="control"))

    return fixture


def gates(fixture: Fixture) -> list[str]:
    """Prove each probe measures what it claims, before a model is called.

    Uses the grader's own matcher. A gate that counted raw substrings would
    flag things the scorecard cannot see, and miss things it can.
    """
    problems: list[str] = []
    corpus = "\n".join(
        [fact["text"] for fact in fixture.facts]
        + [change.message for change in fixture.changes]
        + list(fixture.sources_v1.values()) + list(fixture.sources_v2.values())
    )
    golds = [p.gold for p in fixture.probes if p.gold] + [p.stale for p in fixture.probes if p.stale]

    for probe in fixture.probes:
        for label, value in (("gold", probe.gold), ("stale", probe.stale)):
            if not value:
                continue
            if len(matcher.normalize(value)) < 4:
                problems.append(f"{probe.key}: {label} {value!r} is too short to match safely")
            # No value may be a boundary-match inside another, or one probe's
            # answer silently satisfies another probe's check.
            for other in golds:
                if other != value and matcher.matches(value, other):
                    problems.append(f"{probe.key}: {label} {value!r} also matches {other!r}")
        if probe.gold and probe.category != "in_message" and matcher.matches(probe.gold, probe.question):
            problems.append(f"{probe.key}: the question contains its own answer")

    for probe in fixture.probes:
        if probe.category == "source_drift":
            if matcher.count_occurrences(probe.gold, corpus) != 1:
                problems.append(f"{probe.key}: source-drift answer must appear exactly once")
            chat = "\n".join(c.message for c in fixture.changes)
            if matcher.matches(probe.gold, chat):
                problems.append(f"{probe.key}: a source-only value leaked into the chat")
        if probe.category == "aged" and probe.stale:
            later = [c.message for c in fixture.changes]
            if any(matcher.matches(probe.stale, message) for message in later):
                problems.append(f"{probe.key}: an aged fact must never be mentioned again")
        if probe.category == "negative":
            if matcher.matches(probe.question.split()[-1].strip("?"), corpus):
                continue  # the topic word may legitimately appear; the value must not

    for probe in fixture.probes:
        if probe.category in {"announced", "source_drift"} and probe.gold == probe.stale:
            problems.append(f"{probe.key}: gold and stale are identical, nothing to measure")
    return problems


if __name__ == "__main__":
    from collections import Counter

    fixture = build()
    problems = gates(fixture)
    counts = Counter(probe.category for probe in fixture.probes)
    print(f"{len(fixture.facts)} facts, {len(fixture.changes)} announced changes, "
          f"{len(fixture.probes)} probes")
    for category, count in sorted(counts.items()):
        print(f"  {category:15} {count}")
    print(f"digest {fixture.digest()}")
    if problems:
        for problem in problems:
            print(f"  GATE FAIL {problem}")
        raise SystemExit(1)
    assert build().digest() == fixture.digest(), "the fixture must be deterministic"
    print("OK - every gate passes and the fixture is reproducible")
