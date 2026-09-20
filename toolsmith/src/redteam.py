"""Run the attack corpus at both layers and report what held.

    PYTHONPATH=. uv run python -m src.redteam

Inputs:  src/attacks.py
Outputs: three tables, a block rate, a false-positive rate, and an exit code

Not one API token is spent here, which is the point: the claim that the guard
is code rather than a prompt can be tested for free, as many times as you like,
before any model is involved.

The third round is the one worth reading. It removes the guard entirely and
hands the same bodies straight to the sandbox, which is the only way to see
which layer is actually doing the work - and to see honestly which attacks the
runtime layer does not stop on its own. A control never tested with its partner
removed is a control nobody has measured.
"""

from __future__ import annotations

import hashlib
import sys

from rich.console import Console
from rich.table import Table

from src import attacks, config, guard, sandbox

console = Console()


def _same(got: object, want: object) -> bool:
    if isinstance(want, float) and isinstance(got, (int, float)):
        return abs(float(got) - want) < 0.005
    return got == want


def round_benign() -> list[str]:
    """Ordinary tools must pass the guard and return the right answer.

    Without this round, a guard that refuses everything scores perfectly.
    """
    table = Table(title="round 1 - ordinary tools, which must NOT be blocked")
    for column in ("tool", "guard", "returns", ""):
        table.add_column(column, overflow="fold")
    failures: list[str] = []
    for name, source, kwargs, expect in attacks.BENIGN:
        verdict = guard.guard(source)
        if not verdict.ok:
            failures.append(f"{name}: blocked at {verdict.layer}")
            table.add_row(name, f"[red]{verdict.layer}[/]", verdict.reason[:44],
                          "[red]FALSE POSITIVE[/]")
            continue
        row = sandbox.execute(source, [kwargs])[0][0]
        ok = row.get("ok") and _same(row.get("value"), expect)
        if not ok:
            failures.append(f"{name}: returned {row.get('value', row.get('error'))!r}")
        table.add_row(name, "pass", str(row.get("value", row.get("error")))[:44],
                      "[green]OK[/]" if ok else "[red]WRONG[/]")
    console.print(table)
    return failures


def round_guard() -> tuple[int, list[str]]:
    """Every hostile body through the guard, against its expected layer."""
    table = Table(title="round 2 - the guard alone")
    for column in ("attack", "expected", "actual", ""):
        table.add_column(column, overflow="fold")
    blocked, leaked = 0, []
    for name, source, expected in attacks.HOSTILE:
        verdict = guard.guard(source)
        if verdict.ok:
            if expected != "runtime":
                leaked.append(f"{name}: the guard let it through")
            table.add_row(name, expected, "-",
                          "[green]allowed[/]" if expected == "runtime" else "[red]ALLOWED[/]")
            continue
        blocked += 1
        if expected == "runtime":
            table.add_row(name, expected, verdict.layer, "[green]blocked early[/]")
        elif verdict.layer == expected:
            table.add_row(name, expected, verdict.layer, "[green]blocked[/]")
        else:
            # Still a block, but the reason text is the model's only feedback,
            # so a misleading one is a defect worth seeing.
            table.add_row(name, expected, f"[yellow]{verdict.layer}[/]",
                          "[yellow]WRONG LAYER[/]")
    console.print(table)
    return blocked, leaked


def round_no_guard() -> tuple[int, list[str]]:
    """The same bodies handed straight to the sandbox, guard removed."""
    table = Table(title="round 3 - the guard REMOVED, sandbox alone")
    for column in ("attack", "outcome", "what happened", ""):
        table.add_column(column, overflow="fold")
    contained, escaped = 0, []
    for name, source, _ in attacks.HOSTILE:
        results, raw = sandbox.execute(source, [{"n": 3, "value": "1",
                                                 "prefixes": ["22AAAAA0000A1Z"]}])
        row = results[0]
        if raw.outcome != "ok":
            contained += 1
            table.add_row(name, raw.outcome, f"killed after {raw.seconds:.1f}s",
                          "[green]contained[/]")
        elif not row.get("ok"):
            contained += 1
            table.add_row(name, "error", str(row.get("error"))[:46], "[green]contained[/]")
        else:
            escaped.append(f"{name} -> {str(row.get('value'))[:36]}")
            table.add_row(name, "ok", str(row.get("value"))[:46], "[yellow]RAN[/]")
    console.print(table)
    return contained, escaped


def main() -> int:
    """Run all three rounds, plus a canary, and report once."""
    config.ensure_dirs()
    watched = [config.TASKS_PATH, config.ROOT_DIR / ".env"]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in watched if p.exists()}

    false_positives = round_benign()
    blocked, leaked = round_guard()
    contained, escaped = round_no_guard()
    moved = [p.name for p, digest in before.items()
             if hashlib.sha256(p.read_bytes()).hexdigest() != digest]

    total = len(attacks.HOSTILE)
    console.print(f"\n[bold]block rate[/]       {blocked}/{total} by the guard, "
                  f"{contained}/{total} by the sandbox with the guard removed")
    console.print(f"[bold]false positives[/]  {len(false_positives)}/"
                  f"{len(attacks.BENIGN)} ordinary tools blocked")
    console.print(f"[bold]canaries[/]         "
                  f"{'unchanged' if not moved else 'CHANGED: ' + ', '.join(moved)}")
    for line in escaped:
        console.print(f"  [yellow]runtime alone did not stop[/] {line}")
    for line in false_positives + leaked:
        console.print(f"  [red]{line}[/]")

    # Only two things fail the run: an attack the guard let through, and an
    # ordinary tool it refused. A body the sandbox alone could not stop is
    # reported rather than failed - that is what the guard is for.
    failures = len(false_positives) + len(leaked) + len(moved)
    console.print(f"\n[bold]{'every attack contained' if not failures else f'{failures} FAILURE(S)'}[/]")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
