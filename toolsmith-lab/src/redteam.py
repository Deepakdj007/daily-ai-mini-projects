"""Run the attack corpus at both layers and report what held.

    PYTHONPATH=. uv run python -m src.redteam

Inputs:  src/attacks.py
Outputs: four tables, a block rate, a false-positive rate, and an exit code

Not one API token is spent here, which is the point: the claim that the guard
is code rather than a prompt is the one claim in this project that can be
tested for free, as many times as you like, before any model is involved.

Four rounds, and the third is the one worth reading. Round 1 asks whether the
guard blocks an attack. Round 2 removes the guard entirely and hands the same
bodies to the sandbox, which is the only way to see which layer is actually
doing the work - and to see honestly which attacks the runtime layer does not
stop on its own. A control that has never been tested with its partner removed
is a control nobody has measured.
"""

from __future__ import annotations

import hashlib
import sys

from rich.console import Console
from rich.table import Table

from src import attacks, config, guard, sandbox

console = Console()


def _same(got: object, want: object) -> bool:
    """Compare a JSON round-tripped value with what the tool should return."""
    if isinstance(want, float) and isinstance(got, (int, float)):
        return abs(float(got) - want) < 0.005
    return got == want


def round_benign() -> tuple[int, list[str]]:
    """Ten ordinary tools must pass the guard and return the right answer.

    Without this round, a guard that refuses everything scores perfectly.
    """
    table = Table(title="round 0 - ordinary tools, which must NOT be blocked")
    for column in ("tool", "guard", "sandbox", "returns", ""):
        table.add_column(column, overflow="fold")
    failures: list[str] = []

    for name, source, kwargs, expect in attacks.BENIGN:
        verdict = guard.guard(source)
        if not verdict.ok:
            failures.append(f"{name}: blocked at {verdict.layer}")
            table.add_row(name, f"[red]{verdict.layer}[/]", "-", verdict.reason[:48],
                          "[red]FALSE POSITIVE[/]")
            continue
        results, raw = sandbox.execute(source, [kwargs])
        row = results[0]
        if not row.get("ok"):
            failures.append(f"{name}: {row.get('error', 'no result')}")
            table.add_row(name, "pass", f"[red]{raw.outcome}[/]",
                          str(row.get("error"))[:48], "[red]FAIL[/]")
            continue
        got = row["value"]
        ok = _same(got, expect)
        if not ok:
            failures.append(f"{name}: returned {got!r}, expected {expect!r}")
        table.add_row(name, "pass", f"{raw.seconds:.2f}s", str(got)[:48],
                      "[green]OK[/]" if ok else "[red]WRONG[/]")

    console.print(table)
    return len(attacks.BENIGN) - len(failures), failures


def round_guard() -> tuple[int, list[str]]:
    """Every hostile body through the guard, checked against its expected layer."""
    table = Table(title="round 1 - the guard alone")
    for column in ("attack", "expected", "actual", ""):
        table.add_column(column, overflow="fold")
    blocked, failures = 0, []

    for name, source, expected in attacks.HOSTILE:
        verdict = guard.guard(source)
        if verdict.ok:
            mark = "[green]allowed[/]" if expected == "runtime" else "[red]ALLOWED[/]"
            if expected != "runtime":
                failures.append(f"{name}: the guard let it through")
            table.add_row(name, expected, "-", mark)
            continue
        blocked += 1
        if expected == "runtime":
            # Not a failure. It means the static layer caught something we only
            # promised the runtime layer would, which is a bonus, not a bug.
            table.add_row(name, expected, verdict.layer, "[green]blocked early[/]")
        elif verdict.layer == expected:
            table.add_row(name, expected, verdict.layer, "[green]blocked[/]")
        else:
            # A block at the wrong layer is still a block, but the reason text
            # is the model's only feedback, so a misleading one is a defect.
            failures.append(f"{name}: blocked at {verdict.layer}, expected {expected}")
            table.add_row(name, expected, f"[yellow]{verdict.layer}[/]",
                          "[yellow]WRONG LAYER[/]")

    console.print(table)
    return blocked, failures


def round_no_guard() -> tuple[int, list[str]]:
    """The same bodies handed straight to the sandbox, with the guard removed.

    This is the honest table. Anything marked RAN is an attack the runtime
    layer does not stop on its own - and some of them cannot be stopped there,
    because they never become an operation an audit hook can see.
    """
    table = Table(title="round 2 - the guard REMOVED, sandbox alone")
    for column in ("attack", "outcome", "what happened", ""):
        table.add_column(column, overflow="fold")
    contained, escaped = 0, []

    for name, source, _ in attacks.HOSTILE:
        results, raw = sandbox.execute(source, [{"n": 3, "value": "1",
                                                 "prefix": "1", "number": "1"}])
        row = results[0]
        if raw.outcome != "ok":
            contained += 1
            table.add_row(name, raw.outcome, f"killed after {raw.seconds:.1f}s",
                          "[green]contained[/]")
        elif not row.get("ok"):
            contained += 1
            table.add_row(name, "error", str(row.get("error"))[:52], "[green]contained[/]")
        else:
            escaped.append(f"{name}: ran, returned {str(row.get('value'))[:40]}")
            table.add_row(name, "ok", str(row.get("value"))[:52], "[yellow]RAN[/]")

    console.print(table)
    return contained, escaped


def round_canaries() -> list[str]:
    """Nothing on disk may have moved while the corpus was running."""
    watched = [config.TASKS_PATH, config.ROOT_DIR / "pyproject.toml",
               config.ROOT_DIR / ".env"]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in watched if p.exists()}
    for _, source, _ in attacks.HOSTILE[:12]:
        sandbox.execute(source, [{"n": 1}])
    problems = [f"{path.name} changed during the run"
                for path, digest in before.items()
                if hashlib.sha256(path.read_bytes()).hexdigest() != digest]
    table = Table(title="round 3 - canaries")
    table.add_column("file")
    table.add_column("sha256 before / after")
    table.add_column("")
    for path, digest in before.items():
        now = hashlib.sha256(path.read_bytes()).hexdigest()
        table.add_row(path.name, f"{digest[:12]} / {now[:12]}",
                      "[green]unchanged[/]" if now == digest else "[red]CHANGED[/]")
    console.print(table)
    return problems


def main() -> int:
    """Run all four rounds and report once."""
    config.ensure_dirs()
    passed, false_positives = round_benign()
    blocked, wrong_layer = round_guard()
    contained, escaped = round_no_guard()
    canary_problems = round_canaries()

    total_hostile = len(attacks.HOSTILE)
    console.print()
    console.print(f"[bold]block rate[/]        {blocked}/{total_hostile} by the guard, "
                  f"{contained}/{total_hostile} by the sandbox with the guard removed")
    console.print(f"[bold]false positives[/]   "
                  f"{len(attacks.BENIGN) - passed}/{len(attacks.BENIGN)} ordinary tools blocked")
    for line in escaped:
        console.print(f"  [yellow]runtime alone did not stop[/] {line}")
    for line in false_positives:
        console.print(f"  [red]false positive[/] {line}")
    for line in wrong_layer:
        console.print(f"  [yellow]misleading reason[/] {line}")
    for line in canary_problems:
        console.print(f"  [red]canary[/] {line}")

    # Only two things fail the run. An attack the guard let through, and an
    # ordinary tool it refused. A body the sandbox alone could not stop is
    # reported rather than failed: that is what the guard is for, and a block
    # at an unexpected layer is a wording defect, not a hole.
    leaked = [line for line in wrong_layer if "let it through" in line]
    failures = len(false_positives) + len(leaked) + len(canary_problems)
    console.print(f"\n[bold]{'every attack contained' if not failures else 'FAILURES'}[/]"
                  f"{'' if not failures else f': {failures}'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
