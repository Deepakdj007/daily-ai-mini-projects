"""Turn the results into a scorecard, and check the scorecard is trustworthy.

Inputs:  output/results-<model>.json
Outputs: Rich tables and output/leaderboard.md - at zero API cost

Run this before anything expensive. Every table and both figures rebuild from
the committed results file, so a reader with no daily tokens left still sees
the whole result.

The validity gates run BEFORE the criterion verdict and they void the run.
A scorecard that looks clean is worth nothing if a probe turned out to be
answerable from somewhere the layer did not put it, and the point of checking
first is that a good-looking headline is exactly when nobody looks.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

from rich.console import Console
from rich.table import Table

from src import config
from src.harness import load
from src.ladder import BY_NAME, HEADLINE
from src.types import NON_SCORED

console = Console()

CRITERION = (
    "Retrieval and summarisation earn their place iff, at the same "
    "900-token budget, the full pipeline beats cap+pin+window by at least 20 "
    "points on the out-of-window needles, with exact-McNemar p < 0.01."
)

HEADLINE_ZONES = ("mid", "deep")


def _rows_by_arm(rows: Sequence[dict]) -> dict[str, list[dict]]:
    """Group result rows by arm, preserving order."""
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        out[r["arm"]].append(r)
    return out


def recall(rows: Sequence[dict], zones: Sequence[str] = HEADLINE_ZONES) -> tuple[int, int]:
    """Correct answers over probes attempted, in the given zones.

    Abstentions and service failures both count in the denominator. An arm that
    declined to answer did not recall the fact, and an arm whose requests were
    rejected did not either - scoring only the rows that came back would let
    the arm that failed most often look best.
    """
    hits = [r for r in rows if r["zone"] in zones]
    return sum(1 for r in hits if r["status"] == "correct"), len(hits)


def decomposition(rows: Sequence[dict]) -> dict[str, int]:
    """The 2x2 that says whether the layer failed or the model did."""
    out = {"present_correct": 0, "present_missed": 0,
           "absent_correct": 0, "absent_missed": 0, "not_scored": 0}
    for r in rows:
        if r["zone"] not in HEADLINE_ZONES:
            continue
        if r["status"] in NON_SCORED:
            out["not_scored"] += 1
            continue
        present = r["fact_present"]
        correct = r["status"] == "correct"
        if present is None:
            continue
        key = f"{'present' if present else 'absent'}_{'correct' if correct else 'missed'}"
        out[key] += 1
    return out


def mcnemar(a_rows: Sequence[dict], b_rows: Sequence[dict]) -> tuple[int, int, float]:
    """Exact McNemar on the discordant pairs of two arms.

    Paired on probe id, and both vectors are full length: a dropped row would
    misalign every later pair, so a non-scored outcome counts as a failure for
    pairing and is reported separately.
    """
    from src.stats import mcnemar_exact

    a = {r["probe_id"]: r for r in a_rows if r["zone"] in HEADLINE_ZONES}
    b = {r["probe_id"]: r for r in b_rows if r["zone"] in HEADLINE_ZONES}
    shared = sorted(set(a) & set(b))
    va = [a[p]["status"] == "correct" for p in shared]
    vb = [b[p]["status"] == "correct" for p in shared]
    return mcnemar_exact(va, vb)


def by_carrier(rows: Sequence[dict]) -> dict[str, tuple[int, int]]:
    """Out-of-window recall split by how the fact was written down.

    This turned out to be the sharpest split in the whole run, and it was not
    the one the design was built to measure. A lexical retriever matches
    tokens, so a fact sitting in a JSON field with a distinctive name and
    value is findable; the same fact paraphrased into a sentence is not - and
    the anti-rigging gate that forces low overlap between a probe and the
    transcript is exactly what makes prose invisible to it.
    """
    out: dict[str, list[int]] = {"prose": [0, 0], "tool": [0, 0]}
    for r in rows:
        if r["zone"] not in HEADLINE_ZONES:
            continue
        key = "prose" if r["carrier"] == "prose" else "tool"
        out[key][1] += 1
        out[key][0] += r["status"] == "correct"
    return {k: (v[0], v[1]) for k, v in out.items()}


def validity(rows: Sequence[dict]) -> list[tuple[str, bool, str]]:
    """The checks that void a run. Reported before any verdict."""
    by_arm = _rows_by_arm(rows)
    checks: list[tuple[str, bool, str]] = []

    contaminated = [r for r in rows if r.get("contaminated")]
    checks.append((
        "no contamination",
        not contaminated,
        f"{len(contaminated)} rows answered correctly from a context the grep "
        f"proves the fact was absent from",
    ))

    bad_recent = []
    for arm, arm_rows in by_arm.items():
        if arm in ("none", "raw"):
            continue
        for r in arm_rows:
            if r["zone"] == "recent" and r["status"] != "correct":
                bad_recent.append(f"{arm}/{r['probe_id']}={r['status']}")
    checks.append((
        "recent zone is a clean sweep",
        not bad_recent,
        ", ".join(bad_recent) if bad_recent else
        "every arm answers every fact inside its own window",
    ))

    null_hits = [
        r for r in by_arm.get("none", [])
        if r["status"] == "correct" and r["zone"] in ("recent", "mid", "deep")
    ]
    checks.append((
        "null context scores zero",
        not null_hits,
        ", ".join(r["probe_id"] for r in null_hits) if null_hits else
        "no planted fact is guessable without the conversation",
    ))

    nopin = by_arm.get("no-pin", [])
    pinned_only = [r for r in nopin if r["probe_id"] == "d02"]
    if pinned_only:
        failed = all(r["status"] != "correct" for r in pinned_only)
        checks.append((
            "no-pin control fails the pinned probe",
            failed,
            "the pinned block really is removed when pin is off" if failed else
            "the pinned probe still passes with pin off - pin is not being removed",
        ))

    truncations = sum(1 for r in rows if r["status"] == "truncated")
    rate = truncations / max(len(rows), 1)
    checks.append((
        "truncation under ceiling",
        rate <= config.TRUNCATION_CEILING,
        f"{rate:.1%} of rows lost their answer to the completion cap",
    ))
    return checks


def build_tables(manifest: dict, rows: Sequence[dict]) -> list[Table]:
    """Every table the leaderboard shows."""
    by_arm = _rows_by_arm(rows)
    order = [a for a in ("none", "raw", "window", "cap", "pin", "summarize",
                         "full", "no-pin") if a in by_arm]

    main = Table(title="the ladder", show_header=True, header_style="bold")
    for col in ("arm", "what it adds", "out-of-window recall", "recent",
                "turns held", "assembled tok", "uncached tok", "statuses"):
        main.add_column(col, justify="right" if col != "what it adds" else "left")
    for name in order:
        arm_rows = by_arm[name]
        hit, n = recall(arm_rows)
        rec_hit, rec_n = recall(arm_rows, ("recent",))
        scored = [r for r in arm_rows if r["status"] not in NON_SCORED]
        turns = max((r["turns_in_context"] for r in arm_rows), default=0)
        tok = max((r["assembled_tokens"] for r in arm_rows), default=0)
        unc = sum(r["usage"]["prompt_tokens"] - (r["usage"]["cached_tokens"] or 0)
                  for r in arm_rows)
        counts: dict[str, int] = defaultdict(int)
        for r in arm_rows:
            counts[r["status"]] += 1
        odd = ", ".join(f"{k} {v}" for k, v in sorted(counts.items())
                        if k not in ("correct", "wrong", "abstained")) or "-"
        main.add_row(
            name, BY_NAME[name].label if name in BY_NAME else "",
            f"{hit}/{n}" + (f"  {hit / n:.0%}" if n else ""),
            f"{rec_hit}/{rec_n}" if rec_n else "-",
            str(turns), f"{tok:,}", f"{unc:,}",
            odd if not scored else odd,
        )

    dec = Table(title="did the layer fail, or the model? (out-of-window probes)")
    for col in ("arm", "present & correct", "present & missed",
                "absent & missed", "absent & correct", "not scored"):
        dec.add_column(col, justify="right" if col != "arm" else "left")
    for name in order:
        d = decomposition(by_arm[name])
        dec.add_row(name, str(d["present_correct"]), str(d["present_missed"]),
                    str(d["absent_missed"]), str(d["absent_correct"]),
                    str(d["not_scored"]))

    zones = Table(title="recall by position")
    zones.add_column("arm")
    for z in ("recent", "mid", "deep", "negative", "doc"):
        zones.add_column(z, justify="right")
    for name in order:
        row = [name]
        for z in ("recent", "mid", "deep", "negative", "doc"):
            hit, n = recall(by_arm[name], (z,))
            row.append(f"{hit}/{n}" if n else "-")
        zones.add_row(*row)

    car = Table(title="out-of-window recall by how the fact was written down")
    car.add_column("arm")
    car.add_column("in prose", justify="right")
    car.add_column("in tool output", justify="right")
    for name in order:
        c = by_carrier(by_arm[name])
        prose, tool = c["prose"], c["tool"]
        car.add_row(
            name,
            f"{prose[0]}/{prose[1]}" if prose[1] else "-",
            f"{tool[0]}/{tool[1]}" if tool[1] else "-",
        )

    return [main, dec, zones, car]


def verdict(rows: Sequence[dict]) -> tuple[bool, str]:
    """Evaluate the pre-registered criterion."""
    by_arm = _rows_by_arm(rows)
    a, b = HEADLINE
    if a not in by_arm or b not in by_arm:
        return False, f"{a} or {b} was not run"
    a_hit, a_n = recall(by_arm[a])
    b_hit, b_n = recall(by_arm[b])
    if not a_n or not b_n:
        return False, "no out-of-window probes"
    from src.stats import bootstrap_ci

    gain = (a_hit / a_n - b_hit / b_n) * 100
    only_a, only_b, p = mcnemar(by_arm[a], by_arm[b])
    met = gain >= 20 and p < 0.01
    a_vec = [r["status"] == "correct" for r in by_arm[a] if r["zone"] in HEADLINE_ZONES]
    lo, hi = bootstrap_ci(a_vec)
    return met, (
        f"{a} {a_hit}/{a_n} ({a_hit / a_n:.0%}, 95% CI [{lo:.0%}, {hi:.0%}]) vs "
        f"{b} {b_hit}/{b_n} ({b_hit / b_n:.0%}) = {gain:+.0f} points; "
        f"discordant {only_a}:{only_b}, exact McNemar p = {p:.3f}. "
        f"At n={a_n} one probe is {100 / a_n:.1f} points, so the interval is wide "
        f"and only a gap this large is resolvable."
    )


def main(model: str = "") -> None:
    """Print the scorecard and write the leaderboard."""
    manifest, rows = load(model)
    console.print(f"[bold]{manifest['model']}[/bold]  scenario {manifest['scenario']}  "
                  f"budget {manifest['context_budget']} tok  "
                  f"summary leakage {manifest.get('facts_in_summary')}")

    checks = validity(rows)
    vt = Table(title="validity gates - any failure voids the run")
    vt.add_column("check"); vt.add_column("", justify="center"); vt.add_column("detail")
    for name, ok, detail in checks:
        vt.add_row(name, "[green]pass[/green]" if ok else "[red]FAIL[/red]", detail)
    console.print(vt)

    for table in build_tables(manifest, rows):
        console.print(table)

    met, detail = verdict(rows)
    console.print(f"\n[bold]Pre-registered criterion[/bold]\n  {CRITERION}")
    console.print(f"  [bold]{'MET' if met else 'NOT MET'}[/bold] - {detail}")

    from src.leaderboard import write_markdown

    write_markdown(manifest, rows, checks, met, detail)
    console.print(f"\nwrote {config.LEADERBOARD_PATH.name}")


if __name__ == "__main__":
    main()
