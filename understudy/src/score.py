"""Read what the GPU measured and say what it means.

Inputs:  responses.json, written by the Kaggle notebook
Outputs: two tables and a verdict

The notebook computes numbers only a loaded model can produce and renders no
judgement at all. Every threshold lives here, on a laptop, so that changing
one's mind about what counts as a result does not cost another GPU session.

Style is measured as the negative log-likelihood the model assigns to real
replies it was never trained on. Lower means the reply surprised it less.
This is not the same as generating well, which is why sample generations are
printed underneath the number.

The control is the whole argument. The same measurement runs over a different
person's replies from the same chat. A fine-tune that improves both equally
has learned what WhatsApp looks like, and nothing whatsoever about you.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import exp
from pathlib import Path
from statistics import median

from rich.console import Console
from rich.table import Table

from src.facts import MIN_USABLE_FACTS, Fact
from src.grade import HIT, MISS, NEGATED, SHOTGUN, grade

SCHEMA = "understudy/responses/1"

# How much of your gain the control also got. Compared in nats, not as a
# percentage of each arm's own baseline: the two arms start at different
# losses, so the same nat drop reads as a smaller percentage for whoever
# started worse, and the comparison silently flatters you. That is not a
# hypothetical - it is what the first real run did.
MOSTLY_GENERIC = 0.60
MOSTLY_PERSONAL = 0.35

console = Console()


@dataclass(slots=True)
class Arm:
    """One measured side: the base model, or the tuned one."""

    nll: float
    perplexity: float
    wins: int
    total: int


def load_responses(path: Path) -> dict:
    """Read the notebook's output, refusing anything this file cannot read."""
    if not path.exists():
        raise SystemExit(
            f"{path} is not here. Run the Kaggle notebook, download the "
            f"responses.json it writes, and put it at this path.")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise SystemExit(
            f"{path} says schema {data.get('schema')!r}, this build reads "
            f"{SCHEMA!r}. Re-run the notebook that came with this project.")
    return data


def _weighted_nll(rows: list[dict], key: str) -> float:
    """Mean loss per token, not per reply - one long reply is not one vote."""
    tokens = sum(row["tokens"] for row in rows)
    return sum(row[key] * row["tokens"] for row in rows) / tokens if tokens else 0.0


def measure(rows: list[dict]) -> tuple[Arm, Arm]:
    """Summarise one held-out set under both models."""
    base = _weighted_nll(rows, "nll_base")
    tuned = _weighted_nll(rows, "nll_tuned")
    wins = sum(1 for row in rows if row["nll_tuned"] < row["nll_base"])
    return (Arm(base, exp(base), len(rows) - wins, len(rows)),
            Arm(tuned, exp(tuned), wins, len(rows)))


def drop(base: Arm, tuned: Arm) -> float:
    """How far the loss fell, in nats. The unit the arms can be compared in."""
    return base.nll - tuned.nll


def improvement(base: Arm, tuned: Arm) -> float:
    """The same fall as a percentage of where it started. Display only.

    Never compare two arms with this. A percentage is relative to each arm's
    own baseline, so the arm that started worse is penalised for it.
    """
    return (base.nll - tuned.nll) / base.nll * 100 if base.nll else 0.0


def grade_facts(facts: list[Fact], answers: list[dict]) -> list[dict]:
    """Grade both arms on every fact, and mark the ones the base already knew."""
    # Only gated facts are scored. A question whose answer was never in the
    # training split tests nothing, and counting its miss would make the
    # headline number look better than the run earned.
    by_id = {fact.fact_id: fact for fact in facts if fact.status == "usable"}
    all_golds = [gold for fact in facts for gold in fact.golds]
    graded = []
    for answer in answers:
        fact = by_id.get(answer["id"])
        if fact is None:
            continue
        others = [gold for gold in all_golds if gold not in fact.golds]
        graded.append({
            "fact": fact,
            "base": grade(fact.golds, answer["base"], others),
            "tuned": grade(fact.golds, answer["tuned"], others),
            "base_reply": answer["base"],
            "tuned_reply": answer["tuned"],
        })
    return graded


def show_style(data: dict) -> tuple[float, float, bool]:
    """Print the style tables. Returns both gains and whether the voice moved."""
    you_base, you_tuned = measure(data["style"]["you"])
    their_base, their_tuned = measure(data["style"]["stranger"])
    you_gain, their_gain = drop(you_base, you_tuned), drop(their_base, their_tuned)

    table = Table(title="how surprised the model is by real replies "
                        "(lower is more like that person)", title_justify="left")
    for column in ("whose replies", "base", "tuned", "fell by", "and as %",
                   "replies it fits better"):
        table.add_column(column, justify="right" if column != "whose replies" else "left")
    table.add_row("yours", f"{you_base.nll:.3f}", f"{you_tuned.nll:.3f}",
                  f"[green]{you_gain:.3f}[/green]",
                  f"{improvement(you_base, you_tuned):+.1f}%",
                  f"{you_tuned.wins}/{you_tuned.total}")
    table.add_row(f"{data['run'].get('stranger', 'the other person')} (control)",
                  f"{their_base.nll:.3f}", f"{their_tuned.nll:.3f}",
                  f"{their_gain:.3f}", f"{improvement(their_base, their_tuned):+.1f}%",
                  f"{their_tuned.wins}/{their_tuned.total}")
    console.print(table)
    console.print("[dim]Compare the 'fell by' column, not the percentages - "
                  "the two rows start from different baselines.[/dim]")

    lengths = Table(title="reply length, median characters", title_justify="left",
                    show_header=False)
    gens = data["generations"]
    lengths.add_row("your real replies", f"{median(len(g['real']) for g in gens):.0f}")
    lengths.add_row("base model", f"{median(len(g['base']) for g in gens):.0f}")
    lengths.add_row("tuned model", f"{median(len(g['tuned']) for g in gens):.0f}")
    console.print(lengths)

    for sample in gens[:3]:
        console.print(f"\n[dim]they said[/dim]  {sample['context'][-160:]}")
        console.print(f"[dim]you said [/dim]  {sample['real']}")
        console.print(f"[yellow]base     [/yellow]  {sample['base'][:160]}")
        console.print(f"[green]tuned    [/green]  {sample['tuned'][:160]}")
    return you_gain, their_gain, you_gain > 0 and you_tuned.wins * 2 > you_tuned.total


def show_facts(graded: list[dict]) -> None:
    """Print what each arm recalled, rarest facts last."""
    table = Table(title="facts that are in the training data", title_justify="left")
    for column in ("id", "answer", "said", "base", "tuned"):
        table.add_column(column)
    for row in sorted(graded, key=lambda r: -r["fact"].train_occurrences):
        leaked = row["base"] in (HIT, SHOTGUN)
        table.add_row(
            row["fact"].fact_id, row["fact"].gold,
            str(row["fact"].train_occurrences),
            f"[yellow]{row['base']}[/yellow]" if leaked else row["base"],
            f"[green]{row['tuned']}[/green]" if row["tuned"] == HIT else row["tuned"])
    console.print(table)


def conclude(you_gain: float, their_gain: float, moved: bool,
             graded: list[dict]) -> None:
    """Say what happened, including when what happened is inconclusive."""
    if not moved:
        console.print("\n[red]The voice did not move.[/red] The fine-tune did "
                      "not take, so nothing can be concluded about facts - a "
                      "model that learned nothing forgets nothing. Check that "
                      "the training loss actually fell.")
        return

    console.print(f"\n[green]The voice moved.[/green] Your held-out replies "
                  f"surprise the tuned model {you_gain:.3f} nats less than "
                  f"they surprise the base.")

    share = their_gain / you_gain if you_gain > 0 else 1.0
    if share >= MOSTLY_GENERIC:
        console.print(
            f"[yellow]But most of that is not about you.[/yellow] The same "
            f"training moved someone else's replies by {their_gain:.3f} nats - "
            f"[bold]{share:.0%}[/bold] of your gain. What it mainly learned is "
            f"this conversation: its format, its topics, its names. The part "
            f"that is specifically your voice is the remaining "
            f"{1 - share:.0%}, which is real but is the minority of the effect.")
    elif share >= MOSTLY_PERSONAL:
        console.print(
            f"The control moved {their_gain:.3f} nats, {share:.0%} of your gain. "
            f"So some of this is the chat's general shape and some is you, in "
            f"roughly equal parts. Neither reading on its own is honest.")
    else:
        console.print(
            f"The control moved only {their_gain:.3f} nats, {share:.0%} of your "
            f"gain, so the improvement really is about you rather than about "
            f"WhatsApp in general.")
    console.print("[dim]The control is a hard one: same chat, same topics, "
                  "same format - only the person differs.[/dim]")

    live = [row for row in graded if row["base"] not in (HIT, SHOTGUN)]
    leaked = len(graded) - len(live)
    hits = [row for row in live if row["tuned"] == HIT]
    if leaked:
        console.print(f"\n[dim]{leaked} fact(s) the base model already knew are "
                      f"left out of the count below.[/dim]")
    if len(live) < MIN_USABLE_FACTS:
        console.print(f"[yellow]Only {len(live)} facts left after that. Too few "
                      f"for a percentage - {len(hits)} recalled.[/yellow]")
    else:
        console.print(f"Of {len(live)} facts that are in the training data and "
                      f"that the base model could not answer, the tuned model "
                      f"recalled [bold]{len(hits)}[/bold].")

    if hits:
        repeated = [row for row in hits if row["fact"].train_occurrences > 2]
        console.print("The ones it got were the repeated ones - that is "
                      "memorising a string it saw often, not learning a fact."
                      if len(repeated) * 2 > len(hits) else
                      "Those were not the most repeated ones, so why these and "
                      "not others is not something this run can explain.")
    console.print("\nStyle transferred. Facts did not follow it. For things "
                  "the model has to KNOW, retrieval is the tool - fine-tuning "
                  "is for how it sounds.")


def show_misses(graded: list[dict]) -> None:
    """Print every miss in full, because paraphrase failures flatter the null."""
    console.print("[bold]every fact the tuned model missed, in full[/bold]")
    console.print("[dim]A grader that cannot read a paraphrase scores it as a "
                  "miss, which is the direction that makes this project's point "
                  "for it. Read them.[/dim]\n")
    for row in graded:
        if row["tuned"] in (MISS, NEGATED):
            console.print(f"[cyan]{row['fact'].fact_id}[/cyan] "
                          f"{row['fact'].question}")
            console.print(f"  [dim]wanted[/dim] {row['fact'].gold}")
            console.print(f"  [dim]said  [/dim] {row['tuned_reply'][:300]}\n")


if __name__ == "__main__":
    def rows(count: int, base: float, tuned: float) -> list[dict]:
        return [{"pair_id": f"p{i}", "tokens": 10, "nll_base": base,
                 "nll_tuned": tuned} for i in range(count)]

    base_arm, tuned_arm = measure(rows(10, 3.0, 2.0))
    one_long = measure([{"pair_id": "a", "tokens": 1, "nll_base": 1.0, "nll_tuned": 1.0},
                        {"pair_id": "b", "tokens": 99, "nll_base": 3.0, "nll_tuned": 3.0}])

    checks = [
        ("weighted mean favours the long reply", abs(one_long[0].nll - 2.98) < 0.01),
        ("wins are counted", tuned_arm.wins == 10),
        ("improvement is a percentage", abs(improvement(base_arm, tuned_arm) - 33.3) < 0.1),
        ("drop is in nats", abs(drop(base_arm, tuned_arm) - 1.0) < 1e-9),
        ("no improvement reads as zero",
            improvement(*measure(rows(4, 2.0, 2.0))) == 0.0),
        # The real run's shape: equal nat drops, unequal baselines. Percentages
        # say these arms differ by 17 points; in nats they are identical.
        ("percentages mislead across baselines",   # 50% vs 40%, same 3 nats
            abs(improvement(*measure(rows(4, 6.0, 3.0)))
                - improvement(*measure(rows(4, 7.5, 4.5)))) >= 10
            and abs(drop(*measure(rows(4, 6.0, 3.0)))
                    - drop(*measure(rows(4, 7.5, 4.5)))) < 1e-9),
        ("perplexity is exp of the loss", abs(tuned_arm.perplexity - exp(2.0)) < 1e-9),
        ("an empty set does not divide by zero", _weighted_nll([], "nll_base") == 0.0),
    ]

    fact = Fact("f01", "What is she called?", "Biryani", status="usable")
    fact.train_occurrences = 3
    answer = [{"id": "f01", "base": "I do not know",
               "tuned": "her name is Biryani"}]
    graded = grade_facts([fact], answer)
    rejected = Fact("f02", "Who?", "Raghavan", status="unusable")

    checks += [
        ("tuned hit is graded", graded[0]["tuned"] == HIT),
        ("base miss is graded", graded[0]["base"] == MISS),
        ("unknown fact ids are ignored",
            grade_facts([fact], [{"id": "zz", "base": "", "tuned": ""}]) == []),
        # The notebook may have asked a question the gates threw out; scoring
        # it anyway would pad the denominator with an unanswerable question.
        ("a gated-out fact is not scored",
            grade_facts([rejected], [{"id": "f02", "base": "x",
                                      "tuned": "Raghavan"}]) == []),
    ]
    for name, ok in checks:
        print(f"{'ok  ' if ok else 'FAIL'}  {name}")
    raise SystemExit(0 if all(ok for _, ok in checks) else 1)
