"""The ablation ladder: one new mechanism per rung, machine-checked.

Inputs:  nothing but config defaults
Outputs: LADDER, the arms the harness iterates over

Keeping the experiment's shape in its own file makes one thing checkable at a
glance: every arm differs from its declared parent by exactly one field.
`parent` is a field rather than "the row above" because a linear chain cannot
express a control that hangs off the middle of the ladder.

`pin` is on from the `window` rung upward rather than getting a rung of its
own. It is forty tokens, and a full arm costs about 28,000 - which the measured
cache behaviour says we cannot spend on it. Its control survives as NO_PIN
below, run against the two doc probes only: the pinned-only probe is expected
to FAIL there, and a bug where pin is never actually removed would otherwise
read as a clean null result.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

from src import config
from src.layers.base import PipelineConfig

_BASE = PipelineConfig(context_budget=config.CONTEXT_BUDGET_TOKENS)

OFF = replace(
    _BASE, history=False, cap=False, pin=False,
    retrieve=False, window=False, summarize=False,
)


@dataclass(frozen=True, slots=True)
class Arm:
    """One row of the ladder: a named config plus its provenance."""

    name: str
    label: str
    config: PipelineConfig
    parent: str | None
    changed: str | None
    rationale: str
    probes: str = "all"

    def switch_map(self) -> dict[str, bool]:
        """The on/off switches, for the manifest and the report."""
        return self.config.switch_map()


_none = OFF
_raw = replace(_none, history=True)
_window = replace(_raw, window=True)
_cap = replace(_window, cap=True)
_pinned = replace(_cap, pin=True)
_summ = replace(_pinned, summarize=True)
_full = replace(_summ, retrieve=True)

LADDER: tuple[Arm, ...] = (
    Arm("none", "null context", _none, None, None,
        "Is any probe answerable without the conversation at all? A per-probe "
        "score here is the unguessability check the whole scorecard rests on."),
    Arm("raw", "raw history", _raw, "none", "history",
        "The break. 15,545 tokens against an 8,000-token-per-minute bucket is "
        "rejected outright, not queued - and that wall, not the 131,072-token "
        "context window, is what actually stops a long conversation.",
        probes="sample"),
    Arm("window", "+ window", _window, "raw", "window",
        "The naive fix everyone ships. Survives, and forgets everything older "
        "than a handful of turns."),
    Arm("cap", "+ cap", _cap, "window", "cap",
        "Does shortening fat tool output alone recover the loss? This is the "
        "rung that doubles how many turns fit the same budget."),
    Arm("pin", "+ pin", _pinned, "cap", "pin",
        "Headline baseline. Adds the never-evictable facts block."),
    Arm("summarize", "+ summarize", _summ, "pin", "summarize",
        "The popular fix, and the only layer that rewrites the prefix."),
    Arm("full", "+ retrieve", _full, "summarize", "retrieve",
        "Headline. Does recalling two exact old turns beat the five turns of "
        "recent context it costs to carry them?"),
)

NO_PIN = Arm(
    "no-pin", "full - pin", replace(_full, pin=False), "full", "pin",
    "Ablation control, doc probes only: the pinned-only probe must FAIL here.",
    probes="doc",
)

CONTROLS: tuple[Arm, ...] = (NO_PIN,)
ALL_ARMS: tuple[Arm, ...] = LADDER + CONTROLS
BY_NAME: dict[str, Arm] = {a.name: a for a in ALL_ARMS}

# Which arms each profile runs. The measured cache behaviour decides the
# default: with warm probes costing a full prefix, `full` is a day's budget.
PROFILES: dict[str, tuple[str, ...]] = {
    "smoke": ("none", "raw", "full"),
    # What today's measured cache behaviour actually affords. Keeps both sides
    # of the headline comparison, the null control, the break, the naive floor
    # and the ablation control - and defers the two interior rungs, which the
    # replay cache makes free to add on the next day's refill.
    "lite": ("none", "raw", "window", "pin", "full", "no-pin"),
    "full": tuple(a.name for a in ALL_ARMS),
}

HEADLINE = ("full", "pin")
"""The comparison the criterion is stated over.

Deliberately NOT full-versus-window. The naive window's reach is what defines
the recent zone, so the out-of-window slices sit outside it by construction and
beating it there proves nothing. `pin` (cap+pin+window) is a real alternative
somebody would actually ship, and whether retrieval and summarising beat it is
genuinely uncertain.
"""


def assert_single_switch(arms: tuple[Arm, ...] = ALL_ARMS) -> None:
    """Every arm must differ from its declared parent by exactly one field."""
    by_name = {a.name: a for a in arms}
    for arm in arms:
        if arm.parent is None:
            continue
        parent = by_name.get(arm.parent)
        if parent is None:
            raise AssertionError(f"{arm.name}: unknown parent {arm.parent!r}")
        delta = parent.config.diff(arm.config)
        if delta != (arm.changed,):
            raise AssertionError(
                f"{arm.name}: differs from {arm.parent} in {delta}, "
                f"declared {arm.changed!r}"
            )
        if arm.changed not in {f.name for f in fields(PipelineConfig)}:
            raise AssertionError(f"{arm.name}: {arm.changed!r} is not a real field")


if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table

    assert_single_switch()
    table = Table(title="the ladder - one switch per rung, checked at import")
    for col in ("arm", "parent", "changed", "probes", "hist", "cap", "pin",
                "retr", "win", "summ"):
        table.add_column(col)
    for arm in ALL_ARMS:
        s = arm.switch_map()
        table.add_row(
            arm.name, arm.parent or "-", arm.changed or "-", arm.probes,
            *("on" if s[k] else "." for k in
              ("history", "cap", "pin", "retrieve", "window", "summarize")),
        )
    Console().print(table)
    print(f"headline: {HEADLINE[0]} vs {HEADLINE[1]}")
    print(f"profiles: { {k: len(v) for k, v in PROFILES.items()} }")
