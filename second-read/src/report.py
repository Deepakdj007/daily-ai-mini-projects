"""Reading the results: validity gates first, then the pre-registered verdict.

Inputs:  output/results-<model>.json for the subject and, if present, the twin
Outputs: Rich tables on the terminal and output/leaderboard.md

The gates run before the criterion and can void the run. A good-looking
headline is exactly when nobody checks whether the experiment was valid, so
the order is deliberate. CRITERION is declared here before any run and is not
edited after the numbers arrive.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict

from rich.console import Console
from rich.table import Table

from src import config
from src.audit import LADDER
from src.stats import (fisher_exact, mcnemar_exact, paired_bootstrap_diff, unpaired_bootstrap_diff,
                       wilson_interval)

CRITERION = (
    "The ablation audit earns its place iff, on the authored IU-Xray probes where every "
    "condition completed, the claims it flags as language prior - the model gives the same "
    "answer with the image deleted - are correct at least 15 points less often than the claims "
    "it clears, with Fisher exact p < 0.05, while three bounds hold: blind accuracy sits within "
    "5 points of the 50% floor the probes are balanced to; blind recovery of the minted lab "
    "values is exactly zero; and the image canary reads at least 90% sighted and 0% blind. The "
    "flag rate itself, the gemma3:4b twin and the unmentioned-as-negative shadow set are "
    "reported and are not part of pass or fail."
)
console = Console()
CELLS = ("EARNED", "LUCKY", "PRIOR_ERROR", "MISREAD", "IMAGE_SHAPED", "UNSTABLE", "INCOMPLETE")


def load(model: str) -> dict | None:
    path = config.results_path(model)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _rows(payload: dict, kind: str) -> list[dict]:
    return [r for r in payload["results"] if r["kind"] == kind]


def complete(rows: list[dict]) -> list[dict]:
    """Rows where every condition produced a yes or a no."""
    return [r for r in rows if all(r.get(c) in ("yes", "no") for c in config.CONDITIONS if c in r)
            and not r["truncated"]]


def balanced(rows: list[dict]) -> list[dict]:
    """Equal yes/no per finding after attrition, dropping the last-numbered extras."""
    per: dict[str, dict[str, list]] = defaultdict(lambda: {"yes": [], "no": []})
    for row in sorted(rows, key=lambda r: r["pid"]):
        per[row["finding"]][row["answer"]].append(row)
    out: list[dict] = []
    for sides in per.values():
        n = min(len(sides["yes"]), len(sides["no"]))
        out += sides["yes"][:n] + sides["no"][:n]
    return out


def cell(row: dict) -> str:
    """The named outcome: correctness crossed with the evidence verdict."""
    if row["verdict"] == "unstable":
        return "UNSTABLE"
    if row["verdict"] == "prior":
        return "LUCKY" if row["correct"] else "PRIOR_ERROR"
    if row["verdict"] == "image_shaped":
        return "IMAGE_SHAPED"
    return "EARNED" if row["correct"] else "MISREAD"


def gates(payload: dict) -> list[tuple[str, bool, str]]:
    probes_ = _rows(payload, "probe")
    done = balanced(complete(probes_))
    checks = []
    blind = sum(r["blind"] == r["answer"] for r in done)
    lo, hi = wilson_interval(blind, len(done))
    band = (config.BLIND_FLOOR - config.BLIND_TOLERANCE, config.BLIND_FLOOR + config.BLIND_TOLERANCE)
    checks.append(("blind sits at the 50% construction floor", lo <= band[1] and hi >= band[0],
                   f"{blind}/{len(done)} = {blind / max(1, len(done)):.0%}, 95% CI [{lo:.0%}, {hi:.0%}]"))
    labs = _rows(payload, "lab")
    leaked = sum(r["blind_recovered"] for r in labs)
    total = sum(r["score"]["rows_truth"] for r in labs if r["score"])
    collisions = sum(r.get("blind_prose_collisions", 0) for r in labs)
    checks.append(("blind recovers no minted lab value under its own name", bool(labs) and leaked == 0,
                   f"{leaked} of {total} values over {len(labs)} pages "
                   f"({collisions} loose numeric collisions in blind prose, expected and not a leak)"))
    canary = _rows(payload, "canary")
    seen = sum(r["sighted_read"] for r in canary)
    blind_seen = sum(r["blind_read"] for r in canary)
    checks.append(("image canary: read sighted, silent blind",
                   bool(canary) and seen >= len(canary) * config.CANARY_SIGHTED_FLOOR and blind_seen == 0,
                   f"sighted {seen}/{len(canary)}, blind {blind_seen}/{len(canary)}"))
    bad_bytes = [r["pid"] for r in probes_ if r["image_bytes"].get("sighted", 0) == 0
                 or r["image_bytes"].get("blind", 1) != 0]
    checks.append(("image bytes: >0 on every sighted row, 0 on every blind row", not bad_bytes,
                   "clean" if not bad_bytes else f"{len(bad_bytes)} rows: {bad_bytes[:4]}"))
    incomplete = len(probes_) - len(complete(probes_))
    checks.append(("scored rows completed (no truncation, no unparsed answer)",
                   incomplete <= len(probes_) * 0.05, f"{incomplete}/{len(probes_)} incomplete, excluded"))
    unstable = sum(r["repeat"] != r["sighted"] for r in complete(probes_))
    checks.append(("repeat agrees with sighted (else VOID)",
                   unstable <= len(complete(probes_)) * config.INSTABILITY_CEILING,
                   f"{unstable}/{len(complete(probes_))} disagreements"))
    checks.append(("balance survived attrition", len(done) >= len(complete(probes_)) - 4,
                   f"{len(done)} balanced of {len(complete(probes_))} complete"))
    checks.append(("probe set stamped", str(payload["manifest"].get("probes", "")).startswith("sha256:"),
                   payload["manifest"].get("probes", "missing")[:26]))
    return checks


def verdict(payload: dict, checks: list[tuple[str, bool, str]]) -> tuple[bool, str, list[str]]:
    done = balanced(complete(_rows(payload, "probe")))
    if not done:
        return False, "no complete probe rows", ["run the probes first"]
    flagged = [r for r in done if r["flagged"]]
    cleared = [r for r in done if not r["flagged"]]
    if not flagged or not cleared:
        return False, "the flag never varied", [f"{len(flagged)} flagged, {len(cleared)} cleared - nothing to compare"]
    acc_f = sum(r["correct"] for r in flagged) / len(flagged)
    acc_c = sum(r["correct"] for r in cleared) / len(cleared)
    gap = (acc_c - acc_f) * 100
    lo, hi = unpaired_bootstrap_diff([int(r["correct"]) for r in cleared], [int(r["correct"]) for r in flagged])
    p = fisher_exact(sum(r["correct"] for r in flagged), sum(not r["correct"] for r in flagged),
                     sum(r["correct"] for r in cleared), sum(not r["correct"] for r in cleared))
    wf, wc = wilson_interval(sum(r["correct"] for r in flagged), len(flagged)), \
        wilson_interval(sum(r["correct"] for r in cleared), len(cleared))
    bounds = {name: ok for name, ok, _ in checks}
    detail = [f"{len(flagged)}/{len(done)} claims flagged as language prior ({len(flagged) / len(done):.0%})",
              f"flagged claims correct {acc_f:.0%} (95% CI [{wf[0]:.0%}, {wf[1]:.0%}]); cleared claims correct "
              f"{acc_c:.0%} (95% CI [{wc[0]:.0%}, {wc[1]:.0%}])",
              f"gap {gap:+.0f} points (95% CI {lo * 100:+.0f} to {hi * 100:+.0f}); need >= "
              f"{config.CRITERION_GAP * 100:.0f}",
              f"Fisher exact p = {p:.4f}; need < {config.CRITERION_P}"]
    names = ("blind sits at the 50% construction floor",
             "blind recovers no minted lab value under its own name",
             "image canary: read sighted, silent blind")
    for name in names:
        detail.append(f"bound - {name}: {'holds' if bounds.get(name) else 'FAILS'}")
    met = gap >= config.CRITERION_GAP * 100 and p < config.CRITERION_P and all(bounds.get(n) for n in names)
    return met, f"flagged claims are {gap:+.0f} points less accurate than cleared ones, p = {p:.4f}", detail


def arm_table(payloads: dict[str, dict]) -> Table:
    table = Table(title="accuracy by condition, on the balanced complete probes")
    for col in ("model", "n", "sighted", "blind", "shuffled", "repeat agrees", "flagged (prior)"):
        table.add_column(col, justify="right" if col != "model" else "left")
    for model, payload in payloads.items():
        done = balanced(complete(_rows(payload, "probe")))
        if not done:
            continue
        acc = lambda c: f"{sum(r[c] == r['answer'] for r in done) / len(done):.0%}"  # noqa: E731
        table.add_row(model, str(len(done)), acc("sighted"), acc("blind"), acc("shuffled"),
                      f"{sum(r['repeat'] == r['sighted'] for r in done) / len(done):.0%}",
                      f"{sum(r['flagged'] for r in done) / len(done):.0%}")
    return table


def cell_table(payloads: dict[str, dict]) -> Table:
    table = Table(title="outcome cells - correctness crossed with the evidence verdict")
    table.add_column("model")
    for name in CELLS:
        table.add_column(name, justify="right")
    for model, payload in payloads.items():
        rows = _rows(payload, "probe")
        counts = Counter(cell(r) for r in complete(rows))
        counts["INCOMPLETE"] = len(rows) - len(complete(rows))
        table.add_row(model, *(str(counts[name]) for name in CELLS))
    return table


def finding_table(payload: dict) -> Table:
    table = Table(title="per finding (subject)")
    for col in ("finding", "n", "sighted", "blind", "flagged", "misread"):
        table.add_column(col, justify="right" if col != "finding" else "left")
    per: dict[str, list[dict]] = defaultdict(list)
    for row in balanced(complete(_rows(payload, "probe"))):
        per[row["finding"]].append(row)
    for finding, rows in sorted(per.items()):
        table.add_row(finding, str(len(rows)),
                      f"{sum(r['sighted'] == r['answer'] for r in rows) / len(rows):.0%}",
                      f"{sum(r['blind'] == r['answer'] for r in rows) / len(rows):.0%}",
                      str(sum(r["flagged"] for r in rows)), str(sum(cell(r) == "MISREAD" for r in rows)))
    return table


def secondary(payloads: dict[str, dict]) -> list[str]:
    """Subject minus twin, paired on the same probes; and what the lab pages scored."""
    lines: list[str] = []
    subject = payloads.get(config.SUBJECT_MODEL)
    twin = payloads.get(config.TWIN_MODEL)
    if subject and twin:
        a = {r["pid"]: r for r in complete(_rows(subject, "probe"))}
        b = {r["pid"]: r for r in complete(_rows(twin, "probe"))}
        shared = sorted(set(a) & set(b))
        if shared:
            sa = [a[p]["sighted"] == a[p]["answer"] for p in shared]
            sb = [b[p]["sighted"] == b[p]["answer"] for p in shared]
            gain = (sum(sa) - sum(sb)) / len(shared) * 100
            lo, hi = paired_bootstrap_diff(sa, sb)
            wins, losses, p = mcnemar_exact(sa, sb)
            lines.append(f"sighted: medgemma1.5 {sum(sa) / len(sa):.0%} vs gemma3:4b {sum(sb) / len(sb):.0%} on "
                         f"{len(shared)} shared probes: {gain:+.0f} points (95% CI {lo * 100:+.0f} to "
                         f"{hi * 100:+.0f}), discordant {wins}:{losses}, exact McNemar p = {p:.4f}")
            fa = sum(a[p]["flagged"] for p in shared) / len(shared)
            fb = sum(b[p]["flagged"] for p in shared) / len(shared)
            lines.append(f"language-prior rate: medgemma1.5 {fa:.0%} vs gemma3:4b {fb:.0%} - the difference is "
                         f"what medical tuning installed as prior")
    if subject:
        labs = [r for r in _rows(subject, "lab") if r["score"]]
        if labs:
            acc = sum(r["score"]["field_accuracy"] for r in labs) / len(labs)
            lines.append(f"lab extraction: mean field accuracy {acc:.0%} over {len(labs)} synthetic pages "
                         f"(exact match on eight fields; a ceiling, not a benchmark)")
        shadow = [r for r in _rows(subject, "shadow") if r.get("sighted") in ("yes", "no")]
        if shadow:
            lines.append(f"shadow set (unmentioned counted as no): sighted {sum(r['sighted'] == 'no' for r in shadow) / len(shadow):.0%} "
                         f"'correct' on {len(shadow)} probes - the number the shortcut would have published")
    return lines


def render() -> None:
    payloads = {m: p for m in (config.SUBJECT_MODEL, config.TWIN_MODEL) if (p := load(m))}
    if config.SUBJECT_MODEL not in payloads:
        raise SystemExit(f"no results for {config.SUBJECT_MODEL}; run `python -m src.main run` first")
    subject = payloads[config.SUBJECT_MODEL]
    checks = gates(subject)
    table = Table(title="validity gates - these run before the result, and can void it")
    for col in ("gate", "ok", "detail"):
        table.add_column(col)
    for name, ok, detail in checks:
        table.add_row(name, "[green]yes[/]" if ok else "[red]NO[/]", detail)
    console.print(table)
    console.print(arm_table(payloads))
    console.print(cell_table(payloads))
    console.print(finding_table(subject))
    met, summary, detail = verdict(subject, checks)
    console.print(f"\n[bold]Pre-registered criterion[/]\n{CRITERION}\n")
    for line in detail:
        console.print(f"  {line}")
    extra = secondary(payloads)
    for line in extra:
        console.print(f"  {line}")
    console.print(f"\n[bold]{summary}[/]\n[bold]{'MET' if met else 'NOT MET'}[/]"
                  + ("" if all(ok for _, ok, _ in checks) else "  [red](a gate failed)[/]"))
    _markdown(payloads, checks, met, summary, detail, extra)


def _markdown(payloads, checks, met, summary, detail, extra) -> None:
    manifest = payloads[config.SUBJECT_MODEL]["manifest"]
    lines = ["# second-read leaderboard", "",
             f"Subject `{manifest['model']}` (`{manifest['digest'][:19]}`), profile `{manifest['profile']}`, "
             f"probes `{manifest['probes'][:26]}`, temperature {manifest['temperature']}, top_k "
             f"{manifest['top_k']}, num_ctx {manifest['num_ctx']}, images {manifest['image_edge']}px.", ""]
    lines += ["## Validity gates", "", "| gate | ok | detail |", "|---|:--:|---|"]
    lines += [f"| {n} | {'yes' if ok else '**NO**'} | {d} |" for n, ok, d in checks]
    lines += ["", "## Accuracy by condition", "",
              "| model | n | sighted | blind | shuffled | repeat agrees | flagged |", "|---|--:|--:|--:|--:|--:|--:|"]
    for model, payload in payloads.items():
        done = balanced(complete(_rows(payload, "probe")))
        if done:
            acc = lambda c: f"{sum(r[c] == r['answer'] for r in done) / len(done):.0%}"  # noqa: E731
            lines.append(f"| {model} | {len(done)} | {acc('sighted')} | {acc('blind')} | {acc('shuffled')} | "
                         f"{sum(r['repeat'] == r['sighted'] for r in done) / len(done):.0%} | "
                         f"{sum(r['flagged'] for r in done) / len(done):.0%} |")
    lines += ["", "## Outcome cells", "", "| model | " + " | ".join(CELLS) + " |", "|---|" + "--:|" * len(CELLS)]
    for model, payload in payloads.items():
        rows = _rows(payload, "probe")
        counts = Counter(cell(r) for r in complete(rows))
        counts["INCOMPLETE"] = len(rows) - len(complete(rows))
        lines.append(f"| {model} | " + " | ".join(str(counts[n]) for n in CELLS) + " |")
    lines += ["", "EARNED = correct and grounded. LUCKY = correct but said with the image removed. "
              "PRIOR_ERROR = wrong and said with the image removed. MISREAD = wrong although the image drove "
              "the answer - the audit cannot catch this cell by construction.", ""]
    lines += ["## Pre-registered criterion", "", f"> {CRITERION}", ""] + [f"- {d}" for d in detail]
    lines += ["", f"**{summary}**", "", f"**Result: {'MET' if met else 'NOT MET'}.**", "",
              "The criterion sentence is unchanged from before the run. One instrument behind it was "
              "corrected afterwards: the lab bound originally counted a minted value as recovered if the "
              "number appeared anywhere in the blind answer, which scored two coincidences in a "
              "six-thousand-character essay as leaks. It now requires the value to arrive under its own "
              "analyte name. That correction moves a bound, not the headline - the gap clause fails on "
              "its own numbers either way. See FINDINGS.md #13.", ""]
    if extra:
        lines += ["## Secondary, reported not scored", ""] + [f"- {e}" for e in extra] + [""]
    lines += ["## What each condition is for", ""] + [f"- `{a.name}` ({a.isolates}) predicted: {a.prediction}"
                                                       for a in LADDER]
    config.LEADERBOARD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    render()
