"""The ablation ladder: one new mechanism per rung, checked at import.

Inputs:  nothing but the switches themselves
Outputs: ARMS, the configurations the harness runs

Every arm differs from its declared parent in exactly one field, and
`assert_single_switch()` refuses to let that quietly stop being true. Without
it, `overwrite -> bitemporal` would flip both scope awareness and history at
once and the leaderboard would credit one mechanism for the other's work.

The switches are the same ones the running agent uses. There is no separate
"evaluation mode" of the store, so an arm cannot pass because of something the
demo does not do.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

from src.policy import POLICIES, WritePolicy


@dataclass(frozen=True, slots=True)
class MemoryConfig:
    """Everything that can be switched off, and nothing that cannot."""

    memory: bool = False       # retrieve anything at all
    resolves: bool = False     # ask the classifier at write time
    scope_aware: bool = False  # candidates restricted to a compatible scope
    history: bool = False      # supersede instead of delete, and filter by valid time
    freshness: bool = False    # render the age note, and allow a hedged answer
    sweep: bool = False        # detectors change state durably between turns
    gate: bool = False         # unsure repairs wait for a human

    def diff(self, other: "MemoryConfig") -> tuple[str, ...]:
        return tuple(f.name for f in fields(self)
                     if getattr(self, f.name) != getattr(other, f.name))

    def policy(self) -> WritePolicy:
        """The write policy these switches add up to."""
        if not self.resolves:
            return POLICIES["append"]
        if not self.scope_aware:
            return POLICIES["overwrite"]
        return POLICIES["bitemporal"] if self.history else POLICIES["scoped"]

    def switch_map(self) -> dict[str, bool]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True, slots=True)
class Arm:
    """One row of the ladder: a named config plus why it is there."""

    name: str
    label: str
    config: MemoryConfig
    parent: str | None
    changed: str | None
    rationale: str


OFF = MemoryConfig()
_append = replace(OFF, memory=True)
_overwrite = replace(_append, resolves=True)
_scoped = replace(_overwrite, scope_aware=True)
_bitemporal = replace(_scoped, history=True)
_freshness = replace(_bitemporal, freshness=True)
_sweep = replace(_freshness, sweep=True)
_full = replace(_sweep, gate=True)

LADDER: tuple[Arm, ...] = (
    Arm("none", "no memory", OFF, None, None,
        "Guessability floor. Any probe this arm passes was answerable without "
        "remembering anything, and cannot be evidence for a memory mechanism."),
    Arm("append", "store everything", _append, "none", "memory",
        "The tutorial memory: write every fact, resolve nothing. Contradictions "
        "pile up side by side and retrieval picks whichever embeds closest."),
    Arm("overwrite", "last writer wins", _overwrite, "append", "resolves",
        "The incumbent. A contradiction deletes the old row at write time, which "
        "is what a published memory library does by default."),
    Arm("scoped", "+ scope awareness", _scoped, "overwrite", "scope_aware",
        "Facts qualified by different contexts never meet the classifier, so a "
        "preference that holds at work cannot delete one that holds at home."),
    Arm("bitemporal", "+ keep history", _bitemporal, "scoped", "history",
        "Supersede instead of delete, and filter reads by valid time. The store "
        "a careful engineer would ship."),
    Arm("freshness", "+ show the age", _freshness, "bitemporal", "freshness",
        "The cheap half of staleness: say how old a memory is at read time and "
        "let the model hedge. Costs nothing and needs no background work."),
    Arm("sweep", "+ hygiene sweep", _sweep, "freshness", "sweep",
        "Headline. Durable state changes between turns, and the only rung that "
        "re-reads a source nobody mentioned."),
    Arm("full", "+ human gate", _full, "sweep", "gate",
        "Unsure repairs wait for a person instead of being forced through."),
)

BY_NAME: dict[str, Arm] = {arm.name: arm for arm in LADDER}

HEADLINE = ("sweep", "freshness")
"""The comparison the criterion is stated over.

Deliberately not sweep-versus-bitemporal. Rendering a fact's age is nearly free
and any competent store could do it, so beating a store that does NOT is a low
bar. What is genuinely uncertain is whether a background sweep earns its keep
over simply showing the age - which is what these two arms differ by.
"""

PROFILES: dict[str, tuple[str, ...]] = {
    "smoke": ("none", "overwrite", "sweep"),
    "lite": ("none", "append", "overwrite", "bitemporal", "freshness", "sweep"),
    "full": tuple(arm.name for arm in LADDER),
}


def assert_single_switch(arms: tuple[Arm, ...] = LADDER) -> None:
    """Every arm must differ from its declared parent by exactly one field."""
    names = {arm.name for arm in arms}
    for arm in arms:
        if arm.parent is None:
            continue
        if arm.parent not in names:
            raise AssertionError(f"{arm.name}: unknown parent {arm.parent!r}")
        delta = BY_NAME[arm.parent].config.diff(arm.config)
        if delta != (arm.changed,):
            raise AssertionError(
                f"{arm.name}: differs from {arm.parent} in {delta}, declared {arm.changed!r}"
            )
        if arm.changed not in {f.name for f in fields(MemoryConfig)}:
            raise AssertionError(f"{arm.name}: {arm.changed!r} is not a real switch")


assert_single_switch()


if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table

    switches = [f.name for f in fields(MemoryConfig)]
    table = Table(title="the ladder - one switch per rung, checked at import")
    table.add_column("arm")
    table.add_column("parent")
    table.add_column("changed")
    table.add_column("policy")
    for name in switches:
        table.add_column(name[:5])
    for arm in LADDER:
        flags = arm.config.switch_map()
        table.add_row(arm.name, arm.parent or "-", arm.changed or "-",
                      arm.config.policy().name,
                      *("on" if flags[name] else "." for name in switches))
    Console().print(table)
    print(f"headline: {HEADLINE[0]} vs {HEADLINE[1]}")
    print(f"profiles: { {k: len(v) for k, v in PROFILES.items()} }")
    print("OK - every rung differs from its parent by exactly one switch")
