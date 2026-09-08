"""The ablation ladder: one new mechanism per rung, checked at import.

Inputs:  nothing but the switches themselves
Outputs: LADDER and CONTROLS, the configurations the harness runs

Every arm differs from its declared parent in exactly one field, and
`assert_single_switch()` refuses to let that quietly stop being true. Without
it, a "screening" rung that turned on the classifier, the echo screen and the
tier labels together would let the leaderboard credit one mechanism for
another's work - which is the flaw in the first version of this ladder.

The switches are the ones the running pipeline uses. There is no separate
evaluation mode, so an arm cannot pass because of something the demo does not
do.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace


@dataclass(frozen=True, slots=True)
class Policy:
    """Everything that can be switched off, and nothing that cannot."""

    retrieve: bool = False    # put any retrieved text in front of the model
    rerank: bool = False      # cross-encoder second pass over the dense candidates
    guard: bool = False       # drop passages the injection classifier flags
    echo: bool = False        # drop passages that quote the question back
    isolate: bool = False     # read each passage alone instead of stuffing them together
    resolve: str = "strict"   # strict | majority | provenance

    def diff(self, other: "Policy") -> tuple[str, ...]:
        return tuple(f.name for f in fields(self)
                     if getattr(self, f.name) != getattr(other, f.name))

    def switch_map(self) -> dict[str, object]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True, slots=True)
class Arm:
    """One row of the ladder: a named policy plus why it is there."""

    name: str
    label: str
    policy: Policy
    parent: str | None
    changed: str | None
    rationale: str
    prediction: str = ""
    """Written before the run. A rung whose result was never predicted cannot
    surprise anyone, and a ladder of unsurprising rungs is a demonstration
    rather than a measurement."""


OFF = Policy()
_naive = replace(OFF, retrieve=True)
_rerank = replace(_naive, rerank=True)
_guard = replace(_rerank, guard=True)
_echo = replace(_guard, echo=True)
_isolate = replace(_echo, isolate=True)
_provenance = replace(_isolate, resolve="provenance")
_majority = replace(_isolate, resolve="majority")

LADDER: tuple[Arm, ...] = (
    Arm("none", "no retrieval", OFF, None, None,
        "Guessability floor. Any question this arm answers was answerable "
        "without the corpus, and cannot be evidence for anything below it.",
        "0 of 38. The values are minted, not facts about the world."),
    Arm("naive", "stuff the top 5", _naive, "none", "retrieve",
        "The tutorial pipeline: dense search, five passages into one prompt, "
        "one call. This is what almost every RAG guide ships.",
        "Reproduces PoisonedRAG. Attacker's value asserted on most poisoned "
        "questions, and the model gives no sign anything is wrong."),
    Arm("rerank", "+ cross-encoder", _rerank, "naive", "rerank",
        "The first thing anyone reaches for when retrieval looks wrong. Scores "
        "the question and passage together instead of comparing two vectors.",
        "Helps where two passages in one document are confusable. Does nothing "
        "for poison, and may make it worse: a passage engineered to look "
        "maximally relevant is what a reranker is built to promote."),
    Arm("guard", "+ injection classifier", _guard, "rerank", "guard",
        "Prompt Guard 2 on every passage, scored once at index time on its own "
        "rate-limit bucket. Drops anything over the threshold.",
        "Catches the passage that says 'ignore your instructions'. Blind to the "
        "one that just states a false figure, because that is not an "
        "instruction and there is nothing for a classifier to see."),
    Arm("echo", "+ question-echo screen", _echo, "guard", "echo",
        "Drops passages that repeat the question back in order - the signature "
        "of PoisonedRAG's black-box construction, and of an FAQ.",
        "Kills the verbatim attack outright. Blind to the reordered one. Costs "
        "false positives on the FAQ document, which is why the FAQ is a "
        "separate condition rather than being buried in the clean numbers."),
    Arm("isolate", "+ read passages alone", _isolate, "echo", "isolate",
        "One call per passage, each read with no knowledge of the others, and "
        "every claim verified against the passage that made it. Conflicting "
        "claims produce an abstention.",
        "Attacker's value stops being asserted. So does the right one: on a "
        "poisoned question this abstains, which is safe and not yet useful."),
    Arm("provenance", "+ decide by source", _provenance, "isolate", "resolve",
        "Headline. The conflict is settled in code from the source tier and "
        "document date - metadata the model never sees and a passage cannot "
        "argue with.",
        "Turns isolate's abstentions back into correct answers without "
        "reopening the door, because copies do not count for anything."),
)

CONTROLS: tuple[Arm, ...] = (
    Arm("majority", "+ vote instead", _majority, "isolate", "resolve",
        "The textbook aggregation from the isolate-then-aggregate literature, "
        "run over the identical claim table so it costs no extra calls.",
        "Loses. It assumes many passages independently know the answer, and a "
        "knowledge base keeps each fact in one place, so three copies of a lie "
        "outvote one document telling the truth."),
)

ALL_ARMS: tuple[Arm, ...] = LADDER + CONTROLS
BY_NAME: dict[str, Arm] = {arm.name: arm for arm in ALL_ARMS}

HEADLINE = ("provenance", "isolate")
"""The comparison the criterion is stated over.

Deliberately not provenance-versus-naive. Naive asserts the attacker's value,
so beating it on safety is trivial and proves only that some defence exists.
Isolate already refuses to be fooled; what is genuinely uncertain is whether
anything can recover a CORRECT answer from a poisoned context without becoming
foolable again. That is the gap between these two arms and nowhere else.
"""

PROFILES: dict[str, tuple[str, ...]] = {
    "smoke": ("none", "naive", "isolate", "provenance"),
    "lite": ("none", "naive", "echo", "isolate", "majority", "provenance"),
    "full": tuple(arm.name for arm in ALL_ARMS),
}


def assert_single_switch(arms: tuple[Arm, ...] = ALL_ARMS) -> None:
    """Every arm differs from its parent by exactly one field, and depends sanely."""
    names = {arm.name for arm in arms}
    switches = {f.name for f in fields(Policy)}
    for arm in arms:
        if arm.parent is None:
            continue
        if arm.parent not in names:
            raise AssertionError(f"{arm.name}: unknown parent {arm.parent!r}")
        delta = BY_NAME[arm.parent].policy.diff(arm.policy)
        if delta != (arm.changed,):
            raise AssertionError(
                f"{arm.name}: differs from {arm.parent} in {delta}, declared {arm.changed!r}")
        if arm.changed not in switches:
            raise AssertionError(f"{arm.name}: {arm.changed!r} is not a real switch")
        # Resolution strategies read a claim table, and only isolation builds one.
        if arm.policy.resolve != "strict" and not arm.policy.isolate:
            raise AssertionError(f"{arm.name}: resolve={arm.policy.resolve} needs isolate")


assert_single_switch()


if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table

    switches = [f.name for f in fields(Policy)]
    table = Table(title="the ladder - one switch per rung, checked at import")
    table.add_column("arm")
    table.add_column("parent")
    table.add_column("changed")
    for name in switches:
        table.add_column(name[:6])
    for arm in ALL_ARMS:
        flags = arm.policy.switch_map()
        cells = [str(flags[name]) if name == "resolve"
                 else ("on" if flags[name] else ".") for name in switches]
        style = "cyan" if arm.name in HEADLINE else ""
        table.add_row(arm.name, arm.parent or "-", arm.changed or "-", *cells, style=style)
    Console().print(table)
    print(f"headline: {HEADLINE[0]} vs {HEADLINE[1]}")
    print(f"profiles: { {k: len(v) for k, v in PROFILES.items()} }")
    print("OK - every rung differs from its parent by exactly one switch")
