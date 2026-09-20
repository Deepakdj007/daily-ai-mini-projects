"""The arms, what each one turns on, and what it was predicted to do.

Inputs:  nothing
Outputs: Policy, the arms, the profiles, and an assertion that runs at import

Every rung differs from its parent in exactly one field, which is the only
thing that makes the leaderboard readable: a number that moves between two rows
moved because of one change. assert_single_switch() runs when this module is
imported, so an arm that quietly changes two things cannot reach a run.

The guard is not a rung here. It is on in every arm that writes a tool, and
whether it holds is proved by src/redteam.py, which costs no tokens and can be
re-run as often as anyone likes. Spending a fifth of the daily budget
re-measuring something a free test answers would be a poor trade.

The predictions are written before any run. A rung whose result was never
predicted cannot surprise anyone, and a ladder of unsurprising rungs is a
demonstration rather than a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace


@dataclass(frozen=True, slots=True)
class Policy:
    """Everything that can be switched off, and nothing that cannot."""

    tools: bool = False     # offer the three fixed tools at all
    write: bool = False     # offer write_tool
    test: bool = False      # run the model's own tests before registering
    holdout: bool = False   # run the family's hidden inputs before registering
    persist: bool = False   # the toolbox survives into the next task
    inline: bool = False    # control: run_python, throwaway, nothing registered

    def diff(self, other: "Policy") -> tuple[str, ...]:
        """Which fields differ between two policies."""
        return tuple(f.name for f in fields(self)
                     if getattr(self, f.name) != getattr(other, f.name))


@dataclass(frozen=True, slots=True)
class Arm:
    """One row of the ladder: a named policy, and why it is there."""

    name: str
    label: str
    policy: Policy
    parent: str | None
    changed: str | None
    prediction: str


_NONE = Policy()
_FIXED = replace(_NONE, tools=True)
_WRITE = replace(_FIXED, write=True)
_TESTED = replace(_WRITE, test=True)
_HOLDOUT = replace(_TESTED, holdout=True)
_PERSIST = replace(_HOLDOUT, persist=True)
_INLINE = replace(_FIXED, inline=True)

ALL_ARMS: tuple[Arm, ...] = (
    Arm("none", "no tools at all", _NONE, None, None,
        "0 or 1 of 24. Every family needs a computation it cannot do in its "
        "head. Anything it gets right is a family the gates should have caught."),
    Arm("fixed", "the three fixed tools", _FIXED, "none", "tools",
        "3 to 6 of 24. It reaches for calc and text_stats and they do not "
        "reach far enough. Expect confident wrong check digits rather than "
        "refusals - the interesting failure here is not abstention."),
    Arm("write", "writes tools, nothing checks them", _WRITE, "fixed", "write",
        "12 to 17, and the variance is the story. Untested tools that look "
        "right and are wrong get registered and called, and the model believes "
        "the output. Clearly above fixed, clearly below tested."),
    Arm("tested", "its own tests must pass first", _TESTED, "write", "test",
        "18 to 21. Most of the gain is here rather than at holdout: its own "
        "asserts catch crashes and off-by-ones, which is most of what goes "
        "wrong."),
    Arm("holdout", "hidden inputs must pass too", _HOLDOUT, "tested", "holdout",
        "Within 2 points of tested on first members, better on members 2 to 4. "
        "Predicted to reject at least 2 overfit tools across the six families, "
        "most likely in verhoeff and isoweek. If it rejects none, this rung is "
        "decoration and the guide should say so."),
    Arm("persist", "the toolbox survives the task", _PERSIST, "holdout", "persist",
        "Equal to holdout on correctness within 5 points - a registry does not "
        "make an answer more correct - and 35 to 60% cheaper in completion "
        "tokens on members 2 to 4. Reuse on at least 14 of the 18 repeat tasks."),
    Arm("inline", "throwaway code, nothing kept", _INLINE, "fixed", "inline",
        "Matches persist on correctness and loses on cost: it rewrites the same "
        "algorithm four times per family and pays the repair loop each time, so "
        "its repeat-task spend should sit near persist's FIRST-member cost. "
        "Predicted the most expensive arm in the run."),
)

BY_NAME: dict[str, Arm] = {arm.name: arm for arm in ALL_ARMS}

# The pair the criterion is stated over. Not persist against fixed, which would
# be the flattering comparison: persist versus holdout isolates the registry
# itself, because every other mechanism is already on in both.
HEADLINE = ("persist", "holdout")

# The control predicted to lose, and the reason it might not. If inline turns
# out cheaper, the verdict is NOT MET and the finding is that a registry does
# not pay at this scale - which is a better guide than a rigged win.
CONTROLS = ("none", "inline")

PROFILES: dict[str, tuple[str, ...]] = {
    "smoke": ("none", "fixed", "persist"),
    "lite": ("none", "fixed", "tested", "holdout", "persist"),
    "full": tuple(arm.name for arm in ALL_ARMS),
}


def assert_single_switch(arms: tuple[Arm, ...] = ALL_ARMS) -> None:
    """Every arm differs from its parent by exactly one field, and depends sanely."""
    # Indexed from the arms passed in, not from the module-level table, so the
    # assertion can be pointed at a deliberately broken ladder and shown to fail.
    index = {arm.name: arm for arm in arms}
    names = set(index)
    switches = {f.name for f in fields(Policy)}
    for arm in arms:
        policy = arm.policy
        if arm.policy.test and not policy.write:
            raise AssertionError(f"{arm.name}: test needs write")
        if policy.holdout and not policy.test:
            raise AssertionError(
                f"{arm.name}: holdout without test would run the hidden checks and "
                "not the model's own, which is a mechanism nobody would build")
        if policy.persist and not policy.write:
            raise AssertionError(f"{arm.name}: there is nothing to persist")
        if policy.write and not policy.tools:
            raise AssertionError(f"{arm.name}: write needs tools")
        if policy.inline and policy.write:
            raise AssertionError(
                f"{arm.name}: inline and write are two ways to run model-written "
                "code; an arm with both credits one for the other's work")
        if arm.parent is None:
            continue
        if arm.parent not in names:
            raise AssertionError(f"{arm.name}: unknown parent {arm.parent!r}")
        delta = index[arm.parent].policy.diff(policy)
        if delta != (arm.changed,):
            raise AssertionError(
                f"{arm.name}: differs from {arm.parent} in {delta}, declared "
                f"{arm.changed!r}")
        if arm.changed not in switches:
            raise AssertionError(f"{arm.name}: {arm.changed!r} is not a real switch")


def assert_profiles() -> None:
    """Every profile names arms that exist."""
    for profile, members in PROFILES.items():
        if unknown := set(members) - set(BY_NAME):
            raise AssertionError(f"profile {profile}: unknown arms {unknown}")


assert_single_switch()
assert_profiles()


def arms_for(profile: str) -> tuple[Arm, ...]:
    """The arms in one profile, in ladder order."""
    if profile not in PROFILES:
        raise SystemExit(f"unknown profile {profile!r}; pick one of {sorted(PROFILES)}")
    wanted = set(PROFILES[profile])
    return tuple(arm for arm in ALL_ARMS if arm.name in wanted)


if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="the ladder - one switch per rung")
    for column in ("arm", "changed", *(f.name for f in fields(Policy))):
        table.add_column(column)
    for arm in ALL_ARMS:
        table.add_row(arm.name, arm.changed or "-",
                      *("on" if getattr(arm.policy, f.name) else "."
                        for f in fields(Policy)))
    console.print(table)
    console.print(f"[bold]headline[/]  {HEADLINE[0]} against {HEADLINE[1]}")
    console.print(f"[bold]controls[/]  {', '.join(CONTROLS)}\n")
    for arm in ALL_ARMS:
        console.print(f"[bold]{arm.name}[/] - {arm.label}")
        console.print(f"  {arm.prediction}\n")
    for profile, members in PROFILES.items():
        console.print(f"  {profile:<6} {len(members)} arms: {', '.join(members)}")

    # The assertion has to be able to fail, or it is decoration.
    broken = (Arm("a", "", _FIXED, None, None, ""),
              Arm("b", "", _TESTED, "a", "write", ""))
    try:
        assert_single_switch(broken)
        raise SystemExit("an arm that changes two switches must not pass")
    except AssertionError as exc:
        print(f"\nand it can fail: {exc}")
    print("OK - ladder")
