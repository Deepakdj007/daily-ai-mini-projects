"""Turning a table of claims into one answer, three different ways.

Inputs:  the verified claims from the isolated reads
Outputs: a Verdict - an answer, a hedge, or an explicit abstention

No model runs here. Everything below is arithmetic over claims and over
metadata the generator never saw, which is the whole argument: the decision a
prompt cannot be trusted to make is made somewhere a prompt cannot reach.

Three strategies, and the middle one is in the ladder because it loses.
`majority` is the textbook aggregation from the isolate-then-aggregate
literature, and it is built for open-domain QA where a dozen passages
independently know the answer. An enterprise knowledge base is the opposite
shape: one authoritative passage per fact. Three poisoned copies outvote it.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from src import config, matcher
from src.isolate import Claim


@dataclass(frozen=True, slots=True)
class Verdict:
    """The answer, how it was reached, and what else was on the table."""

    answer: str
    status: str            # answered | hedged | abstained | failed
    cited: str = ""        # the pid the answer came from
    reason: str = ""
    conflicts: tuple[str, ...] = field(default_factory=tuple)

    @property
    def abstained(self) -> bool:
        return self.status == "abstained"


def _group(claims: list[Claim]) -> dict[str, list[Claim]]:
    """Bucket claims by the value they assert, comparing values not strings.

    "Rs 41,904" and "41904" are the same claim. Grouping on the raw text would
    split one agreement into two and hand a disagreement to the resolver.
    """
    groups: dict[str, list[Claim]] = defaultdict(list)
    for claim in claims:
        for key in groups:
            if matcher.matches(key, claim.value) or matcher.matches(claim.value, key):
                groups[key].append(claim)
                break
        else:
            groups[claim.value].append(claim)
    return groups


def _no_claims(claims: list[Claim]) -> Verdict | None:
    """Nothing usable came back. Say which kind of nothing it was."""
    if any(claim.failed for claim in claims):
        return Verdict("", "failed", reason="a passage could not be read")
    return Verdict(matcher.UNKNOWN, "abstained",
                   reason="no retrieved passage states this")


def strict(claims: list[Claim]) -> Verdict:
    """Answer only when every passage that spoke agrees.

    The safe rung. It never asserts the attacker's value, and on a poisoned
    query it never asserts the right one either - an abstention is not an
    answer, and the scorecard counts it as one only where abstaining is the
    correct behaviour.
    """
    usable = [claim for claim in claims if claim.usable]
    if not usable:
        return _no_claims(claims)

    groups = _group(usable)
    if len(groups) == 1:
        value, members = next(iter(groups.items()))
        return Verdict(value, "answered", cited=members[0].pid, reason="all sources agree")
    return Verdict(matcher.UNKNOWN, "abstained", reason="sources disagree",
                   conflicts=tuple(sorted(groups)))


def majority(claims: list[Claim]) -> Verdict:
    """Whichever value the most passages assert. In this shape, the attacker's."""
    usable = [claim for claim in claims if claim.usable]
    if not usable:
        return _no_claims(claims)

    groups = _group(usable)
    ranked = sorted(groups.items(), key=lambda item: -len(item[1]))
    top, members = ranked[0]
    if len(ranked) > 1 and len(ranked[1][1]) == len(members):
        return Verdict(matcher.UNKNOWN, "abstained", reason="tied vote",
                       conflicts=tuple(sorted(groups)))
    return Verdict(top, "answered", cited=members[0].pid,
                   reason=f"{len(members)} of {len(usable)} sources agree",
                   conflicts=tuple(sorted(groups)))


def provenance(claims: list[Claim]) -> Verdict:
    """Highest source tier wins; ties broken by the more recent document.

    Copies do not count for anything here, which is the difference that matters.
    Ten forum posts do not outrank one policy document, so the dose that beats a
    vote changes nothing. What this cannot do is tell a policy document from a
    forged one, and the sametier and embedded conditions are where that shows.
    """
    usable = [claim for claim in claims if claim.usable]
    if not usable:
        return _no_claims(claims)

    groups = _group(usable)

    def authority(item: tuple[str, list[Claim]]) -> tuple[int, str]:
        """A value is as authoritative as the best source asserting it."""
        best = max(item[1], key=lambda claim: (claim.tier, claim.date))
        return best.tier, best.date

    ranked = sorted(groups.items(), key=authority, reverse=True)
    top, members = ranked[0]
    if len(ranked) > 1 and authority(ranked[0]) == authority(ranked[1]):
        return Verdict(matcher.UNKNOWN, "abstained", reason="equally authoritative sources",
                       conflicts=tuple(sorted(groups)))

    winner = max(members, key=lambda claim: (claim.tier, claim.date))
    conflicts = tuple(sorted(groups)) if len(groups) > 1 else ()

    # A claim that only the least trusted tier makes is offered, and flagged.
    # Withholding it would score the same as knowing nothing, and an agent that
    # never repeats what it read from an unverified source is not more useful
    # than one that says so.
    if winner.tier == 0:
        return Verdict(top, "hedged", cited=winner.pid,
                       reason="only an unverified community source says this",
                       conflicts=conflicts)
    if len(groups) == 1:
        return Verdict(top, "answered", cited=winner.pid, reason="all sources agree")
    return Verdict(top, "answered", cited=winner.pid,
                   reason=f"{config.TIER_NAMES[winner.tier]} source, dated {winner.date}",
                   conflicts=conflicts)


STRATEGIES = {"strict": strict, "majority": majority, "provenance": provenance}


if __name__ == "__main__":
    def claim(pid, value, tier, date, verified=True, failed=False):
        return Claim(pid=pid, value=value, verified=verified, tier=tier,
                     date=date, failed=failed)

    # One policy document against three poisoned forum posts: the shape the
    # whole project is about.
    table = [
        claim("gold", "Rs 41,904", 3, "2026-07-14"),
        claim("p1", "Rs 34,980", 0, "2026-07-23"),
        claim("p2", "Rs 34,980", 0, "2026-07-23"),
        claim("p3", "Rs 34,980", 0, "2026-07-23"),
    ]
    assert strict(table).abstained, "disagreement must not produce an answer"
    assert majority(table).answer == "Rs 34,980", "three copies outvote one document"
    assert provenance(table).answer == "Rs 41,904", "tier ignores how many copies there are"

    # Same tier, and the forgery is newer. Recency is all that is left.
    same = [claim("gold", "Rs 41,904", 1, "2026-08-05"),
            claim("fake", "Rs 34,980", 1, "2026-08-26")]
    assert provenance(same).answer == "Rs 34,980", \
        "this is the limit condition, and it is supposed to fail"

    # Only a forum post speaks: offer it, flagged, rather than pretending to know.
    weak = [claim("forum", "Rs 34,980", 0, "2026-07-23")]
    assert provenance(weak).status == "hedged"

    # Values that are the same value spelled differently must not read as conflict.
    spelled = [claim("a", "Rs 41,904", 3, "2026-07-14"), claim("b", "41904", 1, "2026-08-05")]
    assert strict(spelled).status == "answered", "one value, two spellings, no disagreement"

    # A failed read is not an abstention.
    broken = [claim("x", "", 3, "2026-07-14", verified=False, failed=True)]
    assert strict(broken).status == "failed"
    assert strict([]).status == "abstained"
    print("OK - strict abstains, majority loses to the dose, provenance ignores it")
