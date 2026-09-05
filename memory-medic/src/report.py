"""Reading the results file: validity gates first, then the verdict.

Inputs:  output/results-<model>.json
Outputs: Rich tables on the terminal and output/leaderboard.md

The gates run before the criterion and they void the run. A good-looking
headline is exactly when nobody checks whether the experiment was valid, so the
order here is deliberate and not negotiable.

CRITERION is declared at the top of this module, before any run. Quoting a
different comparison after seeing the numbers would be choosing the threshold
to fit the result.
"""

from __future__ import annotations

import json
from collections import defaultdict

from rich.console import Console
from rich.table import Table

from src import config
from src.ladder import BY_NAME, HEADLINE
from src.score import HEADLINE_CATEGORIES
from src.stats import mcnemar_exact, paired_bootstrap_diff, wilson_interval

# Declared before any run. The sweep is compared against showing a fact's age,
# not against a store that hides it, because rendering an age is nearly free and
# beating something that does not do it would prove very little.
CRITERION = (
    "The hygiene sweep earns its place iff, on the source-drift probes, `sweep` "
    "beats `freshness` by at least 25 points with exact McNemar p < 0.01, while "
    "passing at least 90% of the stable-fact controls and losing nothing on the "
    "probes `freshness` already handled."
)

console = Console()


def load(model: str = "") -> dict:
    """Read the committed results file for a model."""
    path = config.results_path(model or config.CHAT_MODEL)
    if not path.exists():
        raise SystemExit(f"no results at {path}. Run `python -m src.main run` first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _by_arm(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["arm"]].append(row)
    return grouped


def gates(payload: dict) -> list[tuple[str, bool, str]]:
    """Every check that can void the run, with the number it is based on."""
    rows = payload["results"]
    grouped = _by_arm(rows)
    checks: list[tuple[str, bool, str]] = []

    leaks = [row for row in rows if row["leaked"]]
    checks.append(("no context leaks", not leaks,
                   f"{len(leaks)} rows where an arm saw a value it should not have"))

    if "none" in grouped:
        planted = [row for row in grouped["none"]
                   if row["category"] not in {"in_message", "negative"} and row["gold"]]
        guessed = [row for row in planted if row["passed"]]
        checks.append(("null arm cannot guess", not guessed,
                       f"{len(guessed)}/{len(planted)} planted probes passed with no memory"))

    controls = [row for row in rows if row["category"] == "in_message"]
    by_arm_control = {arm: [r for r in rs if r["category"] == "in_message"]
                      for arm, rs in grouped.items()}
    worst = min((sum(r["passed"] for r in rs) / len(rs) for rs in by_arm_control.values() if rs),
                default=1.0)
    checks.append(("controls answerable in every arm", worst >= 0.66,
                   f"worst arm passes {worst:.0%} of the in-message controls"))

    negatives = [row for row in rows if row["category"] == "negative"]
    confabulated = [row for row in negatives if not row["passed"]]
    checks.append(("no confabulation on unknowns", len(confabulated) <= len(negatives) * 0.1,
                   f"{len(confabulated)}/{len(negatives)} answered a question never discussed"))

    failed = payload["manifest"].get("llm_calls", {}).get("failed", 0)
    checks.append(("no failed model calls", failed == 0, f"{failed} calls failed"))

    fixture_stamped = bool(payload["manifest"].get("fixture", "").startswith("sha256:"))
    checks.append(("fixture is stamped", fixture_stamped,
                   payload["manifest"].get("fixture", "missing")))
    return checks


def category_table(payload: dict) -> Table:
    """Pass rate per arm per category - the diagonal, or the absence of one."""
    rows = payload["results"]
    grouped = _by_arm(rows)
    categories = sorted({row["category"] for row in rows})
    table = Table(title="probes passed, by arm and category")
    table.add_column("arm")
    for category in categories:
        table.add_column(category.replace("_", " ")[:12])
    for arm in payload["manifest"]["arms"]:
        cells = []
        for category in categories:
            subset = [r for r in grouped.get(arm, []) if r["category"] == category]
            passed = sum(r["passed"] for r in subset)
            style = "green" if subset and passed == len(subset) else (
                "red" if subset and passed == 0 else "yellow")
            cells.append(f"[{style}]{passed}/{len(subset)}[/]" if subset else "-")
        table.add_row(arm, *cells)
    return table


def check_table(payload: dict) -> Table:
    """Per-check pass rates, so one number never hides two behaviours."""
    grouped = _by_arm(payload["results"])
    tally: dict[tuple[str, str], list[int]] = defaultdict(list)
    for arm, rows in grouped.items():
        for row in rows:
            for name, ok in row["checks"].items():
                tally[(arm, name)].append(int(ok))
    names = sorted({name for _, name in tally})
    table = Table(title="individual checks (a probe passes only when all of its checks do)")
    table.add_column("check")
    for arm in payload["manifest"]["arms"]:
        table.add_column(arm[:10])
    for name in names:
        cells = []
        for arm in payload["manifest"]["arms"]:
            values = tally.get((arm, name), [])
            cells.append(f"{sum(values)}/{len(values)}" if values else "-")
        table.add_row(name[:44], *cells)
    return table


def verdict(payload: dict) -> tuple[bool, str, dict]:
    """Test the pre-registered criterion against the headline arms."""
    treatment, baseline = HEADLINE
    grouped = _by_arm(payload["results"])
    if treatment not in grouped or baseline not in grouped:
        return False, f"the headline arms ({treatment} vs {baseline}) were not both run", {}

    def outcomes(arm: str, categories) -> list[int]:
        rows = sorted((r for r in grouped[arm] if r["category"] in categories),
                      key=lambda r: r["probe"])
        return [int(r["passed"]) for r in rows]

    after = outcomes(treatment, HEADLINE_CATEGORIES)
    before = outcomes(baseline, HEADLINE_CATEGORIES)
    n = len(after)
    gap = (sum(after) - sum(before)) / n if n else 0.0
    b_count, c_count, p_value = mcnemar_exact(after, before)
    low, high = paired_bootstrap_diff(after, before)

    stable_after = outcomes(treatment, ("stable_control",))
    stable_rate = sum(stable_after) / len(stable_after) if stable_after else 0.0

    kept = outcomes(treatment, ("aged", "announced", "scope_pair"))
    kept_before = outcomes(baseline, ("aged", "announced", "scope_pair"))
    regression = sum(kept_before) - sum(kept)

    met = gap >= 0.25 and p_value < 0.01 and stable_rate >= 0.90 and regression <= 1
    detail = {
        "n": n, "treatment": sum(after), "baseline": sum(before), "gap": gap,
        "p": p_value, "b": b_count, "c": c_count, "ci": (low, high),
        "stable_rate": stable_rate, "regression": regression,
    }
    summary = (
        f"{treatment} {sum(after)}/{n} vs {baseline} {sum(before)}/{n} on source drift: "
        f"{gap:+.0%} (95% CI {low:+.0%} to {high:+.0%}), exact McNemar p={p_value:.4g} "
        f"[{b_count} vs {c_count} discordant]; stable controls {stable_rate:.0%}; "
        f"regression elsewhere {regression}"
    )
    return met, summary, detail


def render(model: str = "") -> None:
    """Print the whole report, gates first."""
    payload = load(model)
    manifest = payload["manifest"]
    console.rule("[bold]memory medic - silent decay leaderboard")
    console.print(f"[dim]{manifest['date']} · {manifest['answer_model']} · "
                  f"fixture {manifest['fixture']} · probes asked as of {manifest['probe_at']}[/]")

    console.print("\n[bold]Validity gates[/] (these run before the verdict, and void it)")
    all_pass = True
    for name, ok, detail in gates(payload):
        all_pass &= ok
        console.print(f"  {'[green]PASS[/]' if ok else '[red]FAIL[/]'} {name:38} [dim]{detail}[/]")

    console.print()
    console.print(category_table(payload))
    console.print()
    console.print(check_table(payload))

    met, summary, _ = verdict(payload)
    console.print(f"\n[bold]Pre-registered criterion[/]\n[dim]{CRITERION}[/]")
    if not all_pass:
        console.print("\n[red]RUN VOID[/] - a validity gate failed, so the verdict below means nothing.")
    console.print(f"\n[bold]{'MET' if met else 'NOT MET'}[/] - {summary}")
    write_leaderboard(payload, met, summary, all_pass)
    console.print(f"\n[dim]written to {config.LEADERBOARD_PATH}[/]")


def write_leaderboard(payload: dict, met: bool, summary: str, gates_pass: bool) -> None:
    """Render the same thing as markdown, for the repository."""
    manifest = payload["manifest"]
    grouped = _by_arm(payload["results"])
    categories = sorted({row["category"] for row in payload["results"]})
    lines = [
        "# memory medic - silent decay leaderboard", "",
        f"> {CRITERION}", "",
        f"**{'MET' if met else 'NOT MET'}** - {summary}", "",
        "" if gates_pass else "> **RUN VOID** - a validity gate failed.\n",
        f"Run {manifest['date']} · answers `{manifest['answer_model']}` · "
        f"extraction `{manifest['extract_model']}` · fixture `{manifest['fixture']}` · "
        f"probes asked as of {manifest['probe_at']} · "
        f"{manifest['tokens_this_run']:,} tokens.", "",
        "## Validity gates", "",
        "| gate | result | detail |", "| --- | --- | --- |",
    ]
    for name, ok, detail in gates(payload):
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")

    lines += ["", "## Probes passed, by arm and category", "",
              "| arm | " + " | ".join(c.replace('_', ' ') for c in categories) + " |",
              "| --- | " + " | ".join("---" for _ in categories) + " |"]
    for arm in manifest["arms"]:
        cells = []
        for category in categories:
            subset = [r for r in grouped.get(arm, []) if r["category"] == category]
            cells.append(f"{sum(r['passed'] for r in subset)}/{len(subset)}" if subset else "-")
        lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")

    lines += ["", "## What each rung adds", "",
              "| arm | switch | why it is here |", "| --- | --- | --- |"]
    for arm in manifest["arms"]:
        spec = BY_NAME[arm]
        lines.append(f"| `{arm}` | {spec.changed or '-'} | {spec.rationale} |")
    lines.append("")
    config.LEADERBOARD_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    render()
