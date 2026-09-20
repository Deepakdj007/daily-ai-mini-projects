"""The scorecard: validity gates first, then the tables, then the verdict.

    PYTHONPATH=. uv run python -m src.report

Inputs:  output/results-<model>.json
Outputs: the tables, MET or NOT MET, and output/leaderboard.md

The order is not negotiable. Gates print before results, because a
good-looking headline is exactly the moment nobody checks whether the run was
valid. A failed gate voids the verdict rather than footnoting it.

The criterion below was committed before any of these numbers existed, and
tools/check_report.py exercises it against synthetic results so that it can be
shown to separate a real effect from a small one. Quoting a different operating
point after seeing a leaderboard is choosing the threshold to fit the result,
and from the outside there is no way to tell the difference afterwards.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter

from rich.console import Console
from rich.table import Table

from src import cache, config, corpus, ladder, stats

console = Console()

# Declared before any run. Stated on the pair that isolates the registry -
# persist against holdout - and not on persist against fixed, which would be
# the flattering comparison: every other mechanism is already on in both, so
# anything that moves between them moved because the toolbox survived.
#
# It is deliberately not stated on correctness alone. A registry cannot make an
# answer more correct; the claim it is entitled to make is about cost, and a
# claim about cost is only worth anything if correctness is held level and the
# saving is not simply an arm that gave up sooner.
CRITERION = (
    "A persistent tool registry earns its place iff, on the repeat tasks - "
    "members 2 to 4 of each family, where a tool written earlier could be "
    "reused - `persist` is no worse than `holdout` on exact-match correctness, "
    "at most 5 points lower with the 95% paired bootstrap interval on the "
    "difference excluding a 15-point loss; AND spends at most 0.60x `holdout`'s "
    "MEDIAN completion tokens on those tasks, with the 95% paired bootstrap "
    "upper bound on that ratio below 0.85; AND answers at least 70% of them "
    "using a tool it wrote on an EARLIER task, Wilson lower bound at least "
    "0.50. Two further conditions, both able to fail on their own. The `inline` "
    "control, which writes throwaway code every task and keeps nothing, must "
    "spend strictly more median completion tokens on those same repeat tasks "
    "than `persist` does - if four lines of run_python are cheaper than a "
    "registry, the registry has earned nothing. And on the FIRST member of each "
    "family, where there is nothing to reuse and the two arms run identical "
    "code, `persist` and `holdout` must agree within 5 points; a gap there "
    "means the arms differ for a reason the mechanism cannot explain, and voids "
    "the comparison."
)


def load(model: str = "") -> dict:
    """The results file, or a message saying how to make one."""
    path = config.results_path(model)
    if not path.exists():
        raise SystemExit(f"{path} does not exist. Run:\n"
                         "  PYTHONPATH=. uv run python -m src.main run --profile smoke")
    return json.loads(path.read_text(encoding="utf-8"))


def rows_for(payload: dict, *, arm: str = "", member: str = "") -> list[dict]:
    """Rows filtered by arm, and by whether a task is a family's first or a repeat."""
    rows = payload["results"]
    if arm:
        rows = [row for row in rows if row["arm"] == arm]
    if member == "first":
        rows = [row for row in rows if row["member"] == 1]
    elif member == "repeat":
        rows = [row for row in rows if row["member"] > 1]
    return rows


def _paired(payload: dict, left: str, right: str, member: str = "") -> tuple[list, list]:
    """Two arms' rows over the tasks they both ran, in the same order."""
    a = {row["task_id"]: row for row in rows_for(payload, arm=left, member=member)}
    b = {row["task_id"]: row for row in rows_for(payload, arm=right, member=member)}
    shared = [t for t in corpus.load()["task_order"] if t in a and t in b]
    return [a[t] for t in shared], [b[t] for t in shared]


def _rate(rows: list[dict]) -> float:
    return sum(row["correct"] for row in rows) / len(rows) if rows else 0.0


def gates(payload: dict) -> list[tuple[str, bool, str]]:
    """Every check that can void the run, with the number it is based on."""
    rows = payload["results"]
    arms = {row["arm"] for row in rows}
    checks: list[tuple[str, bool, str]] = []

    null = rows_for(payload, arm="none")
    if null:
        guessed = sum(row["correct"] for row in null)
        checks.append(("the null arm cannot guess", guessed <= max(1, len(null) * 0.10),
                       f"{guessed}/{len(null)} right with no tools at all"))

    fixed = rows_for(payload, arm="fixed")
    if fixed:
        by_family = Counter(row["family"] for row in fixed if row["correct"])
        worst = max(by_family.values(), default=0)
        checks.append(("the fixed toolset cannot solve a family",
                       _rate(fixed) <= 0.35 and worst < 3,
                       f"{_rate(fixed):.0%} overall, best family {worst}/4"))

    writer = "tested" if "tested" in arms else ("holdout" if "holdout" in arms else "")
    if writer and fixed:
        left, right = _paired(payload, writer, "fixed")
        gain = (_rate(left) - _rate(right)) * 100
        checks.append(("writing tools helps at all", gain >= 25,
                       f"{writer} beats fixed by {gain:+.0f} points"))

    if "holdout" in arms:
        held = rows_for(payload, arm="holdout")
        fired = [row for row in held if row["rejections"].get("holdout")]
        families = {row["family"] for row in fired}
        checks.append(("the held-out check is not decoration",
                       len(fired) >= 2 and len(families) >= 2,
                       f"{len(fired)} tool(s) rejected by hidden inputs across "
                       f"{len(families)} famil(ies)"))

    leaked = [row["task_id"] for row in rows
              if not ladder.BY_NAME[row["arm"]].policy.persist
              and row["offered"] and any(name not in ("calc", "today", "text_stats",
                                                      "write_tool", "run_python",
                                                      "submit_answer")
                                         for name in row["offered"][0])]
    checks.append(("the toolbox did not leak between tasks", not leaked,
                   f"{len(leaked)} non-persist task(s) started with a written tool"))

    order = {arm: tuple(row["task_id"] for row in rows_for(payload, arm=arm))
             for arm in arms}
    lengths = {len(seq) for seq in order.values()}
    checks.append(("every arm ran the same tasks in the same order",
                   len(set(order.values())) == 1,
                   f"{len(order)} arm(s), {lengths} task(s) each"))

    retried = [row for row in rows if row.get("retried")]
    checks.append(("temperature stayed at zero", len(retried) <= len(rows) * 0.05,
                   f"{len(retried)}/{len(rows)} rows used a non-zero-temperature retry"))

    errors = [row for row in rows if row["outcome"] == "error"]
    checks.append(("no scored row is a failed model call",
                   len(errors) <= len(rows) * 0.05,
                   f"{len(errors)}/{len(rows)} rows ended in an API failure"))

    manifest = payload.get("manifest", {})
    fresh = corpus.stamp()
    checks.append(("the corpus and oracle are stamped and current",
                   manifest.get("corpus") == fresh["corpus"]
                   and manifest.get("oracle") == fresh["oracle"],
                   f"{manifest.get('corpus', 'missing')} / {manifest.get('oracle', '?')}"))
    return checks


def verdict(payload: dict) -> tuple[bool, str, list[str]]:
    """Test the pre-registered criterion. Returns (met, one line, the detail)."""
    arms = {row["arm"] for row in payload["results"]}
    if not {"persist", "holdout"} <= arms:
        return False, "the headline pair was not run", [
            f"needs persist and holdout; this run has {', '.join(sorted(arms))}"]

    detail: list[str] = []
    persist, holdout = _paired(payload, "persist", "holdout", member="repeat")
    if not persist:
        return False, "no repeat tasks in this run", ["members 2 to 4 were not run"]

    gap = (_rate(persist) - _rate(holdout)) * 100
    low, high = stats.paired_bootstrap_diff([int(r["correct"]) for r in persist],
                                            [int(r["correct"]) for r in holdout])
    detail.append(f"correct on {len(persist)} repeat tasks: persist "
                  f"{_rate(persist):.0%} vs holdout {_rate(holdout):.0%} "
                  f"({gap:+.0f} points, 95% CI {low * 100:+.0f} to {high * 100:+.0f})")

    ratio, r_low, r_high = stats.median_ratio_bootstrap(
        [r["completion_tokens"] for r in persist],
        [r["completion_tokens"] for r in holdout])
    detail.append(f"median completion tokens: persist "
                  f"{statistics.median([r['completion_tokens'] for r in persist]):.0f} vs "
                  f"holdout {statistics.median([r['completion_tokens'] for r in holdout]):.0f}"
                  f" = {ratio:.2f}x (95% CI {r_low:.2f} to {r_high:.2f})")

    reused = sum(row["reused"] for row in persist)
    reuse_rate = reused / len(persist)
    w_low, _ = stats.wilson_interval(reused, len(persist))
    detail.append(f"reuse: {reused}/{len(persist)} repeat tasks answered with a tool "
                  f"written earlier ({reuse_rate:.0%}, Wilson low {w_low:.2f})")

    first_p, first_h = _paired(payload, "persist", "holdout", member="first")
    first_gap = abs(_rate(first_p) - _rate(first_h)) * 100 if first_p else 0.0
    detail.append(f"first members, where nothing can be reused: persist "
                  f"{_rate(first_p):.0%} vs holdout {_rate(first_h):.0%} "
                  f"({first_gap:.0f} points apart)")

    inline_ok, inline_note = True, "inline was not run"
    if "inline" in arms:
        inline, mirror = _paired(payload, "inline", "persist", member="repeat")
        inline_median = statistics.median([r["completion_tokens"] for r in inline] or [0])
        persist_median = statistics.median([r["completion_tokens"] for r in mirror] or [0])
        inline_ok = inline_median > persist_median
        inline_note = (f"control: inline spends {inline_median:.0f} median completion "
                       f"tokens per repeat task against persist's {persist_median:.0f}")
    detail.append(inline_note)

    met = (gap >= -5 and low * 100 > -15 and ratio <= 0.60 and r_high < 0.85
           and reuse_rate >= 0.70 and w_low >= 0.50 and first_gap <= 5 and inline_ok)
    summary = (f"persist answers {gap:+.0f} points differently and costs {ratio:.2f}x "
               f"holdout on repeat tasks, reusing on {reuse_rate:.0%} of them")
    return met, summary, detail


def _arm_table(payload: dict) -> Table:
    """Correct, cost and writing behaviour, one row per arm."""
    table = Table(title="by arm")
    for column in ("arm", "correct", "first", "repeat", "median completion",
                   "wrote", "registered", "reused"):
        table.add_column(column, justify="right" if column != "arm" else "left")
    for arm in ladder.ALL_ARMS:
        rows = rows_for(payload, arm=arm.name)
        if not rows:
            continue
        repeat = rows_for(payload, arm=arm.name, member="repeat")
        table.add_row(
            arm.name,
            f"{sum(r['correct'] for r in rows)}/{len(rows)}",
            f"{_rate(rows_for(payload, arm=arm.name, member='first')):.0%}",
            f"{_rate(repeat):.0%}",
            f"{statistics.median([r['completion_tokens'] for r in rows]):.0f}",
            str(sum(r["writes"] for r in rows)),
            str(sum(r["registered"] for r in rows)),
            str(sum(r["reused"] for r in rows)))
    return table


def _ledger_table(payload: dict) -> Table:
    """Why proposed tools were turned away. The table that shows the guard working."""
    stages = ("guard", "hardcode", "selftest", "holdout", "name", "collision",
              "parse", "contract", "crash", "timeout", "refused")
    table = Table(title="the write ledger - why a proposed tool was turned away")
    table.add_column("arm")
    for stage in stages:
        table.add_column(stage, justify="right")
    for arm in ladder.ALL_ARMS:
        rows = rows_for(payload, arm=arm.name)
        if not rows or not any(r["writes"] for r in rows):
            continue
        counts = Counter()
        for row in rows:
            counts.update(row["rejections"])
        table.add_row(arm.name, *(str(counts.get(stage, 0)) or "." for stage in stages))
    return table


def _family_table(payload: dict) -> Table:
    """Correct by family, so a family nobody can do is visible rather than averaged."""
    families = [entry["family"] for entry in corpus.families()]
    table = Table(title="correct by family")
    table.add_column("arm")
    for family in families:
        table.add_column(family, justify="right")
    for arm in ladder.ALL_ARMS:
        rows = rows_for(payload, arm=arm.name)
        if not rows:
            continue
        cells = []
        for family in families:
            here = [r for r in rows if r["family"] == family]
            cells.append(f"{sum(r['correct'] for r in here)}/{len(here)}" if here else "-")
        table.add_row(arm.name, *cells)
    return table


def _write_markdown(payload: dict, checks, met: bool, summary: str,
                    detail: list[str]) -> None:
    """The leaderboard, as a file somebody can read without running anything."""
    manifest = payload.get("manifest", {})
    lines = ["# toolsmith - leaderboard", "",
             f"- model: `{manifest.get('model', '?')}`",
             f"- profile: `{manifest.get('profile', '?')}`",
             f"- corpus: `{manifest.get('corpus', '?')}`",
             f"- oracle: `{manifest.get('oracle', '?')}`",
             f"- rows: {len(payload['results'])}", ""]
    lines += ["## Validity gates", ""]
    for name, ok, note in checks:
        lines.append(f"- {'PASS' if ok else 'FAIL'} - {name} ({note})")
    lines += ["", "## By arm", "",
              "| arm | correct | first | repeat | median completion | wrote | registered | reused |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for arm in ladder.ALL_ARMS:
        rows = rows_for(payload, arm=arm.name)
        if not rows:
            continue
        lines.append(
            f"| {arm.name} | {sum(r['correct'] for r in rows)}/{len(rows)} "
            f"| {_rate(rows_for(payload, arm=arm.name, member='first')):.0%} "
            f"| {_rate(rows_for(payload, arm=arm.name, member='repeat')):.0%} "
            f"| {statistics.median([r['completion_tokens'] for r in rows]):.0f} "
            f"| {sum(r['writes'] for r in rows)} | {sum(r['registered'] for r in rows)} "
            f"| {sum(r['reused'] for r in rows)} |")
    lines += ["", "## Pre-registered criterion", "", CRITERION, ""]
    lines += [f"- {line}" for line in detail]
    lines += ["", f"**Result: {'MET' if met else 'NOT MET'}.** {summary}", "",
              "## What each arm was predicted to do", ""]
    for arm in ladder.ALL_ARMS:
        if rows_for(payload, arm=arm.name):
            lines.append(f"- **{arm.name}** - {arm.prediction}")
    config.LEADERBOARD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render(model: str = "") -> dict:
    """Print gates, tables and the verdict, and write the leaderboard."""
    payload = load(model)
    checks = gates(payload)

    table = Table(title="validity gates - these run before the result, and void it")
    table.add_column("gate")
    table.add_column("")
    table.add_column("detail", overflow="fold")
    for name, ok, note in checks:
        table.add_row(name, "[green]pass[/]" if ok else "[red]FAIL[/]", note)
    console.print(table)

    console.print(_arm_table(payload))
    console.print(_family_table(payload))
    console.print(_ledger_table(payload))

    met, summary, detail = verdict(payload)
    console.print(f"\n[bold]Pre-registered criterion[/]\n{CRITERION}\n")
    for line in detail:
        console.print(f"  {line}")
    console.print(f"\n[bold]{summary}[/]")
    console.print(f"[bold]{'MET' if met else 'NOT MET'}[/]"
                  + ("" if all(ok for _, ok, _ in checks) else "  [red](gates failed)[/]"))
    _write_markdown(payload, checks, met, summary, detail)
    console.print(f"\nwritten to {config.LEADERBOARD_PATH.relative_to(config.ROOT_DIR)}")
    spent, requests = cache.spent_today()
    console.print(f"spent today: {spent:,} tokens over {requests} requests")
    return payload


if __name__ == "__main__":
    render()
