"""The command line: models, build, gates, note, case, locate, compare, ct, ladder, run, report.

Inputs:  argv
Outputs: terminal output and the files under output/

`case` is the one to run first. It writes the note, lifts each claim, re-asks
it four ways and prints the whole trail for one film. Every number the
leaderboard reports is visible there for one case, which is the only way to
believe the aggregate.
"""

from __future__ import annotations

import argparse
import time

from rich.console import Console
from rich.table import Table

from src import config

console = Console()


def cmd_models(args) -> None:
    """Preflight: tags resolve, digests recorded, warm seconds per call."""
    from src import images, llm

    for tag in (config.SUBJECT_MODEL, config.TWIN_MODEL):
        console.print(f"{tag:22s} {llm.digest(tag)}")
    frame = images.burn_text("WARM")
    llm.complete(config.SUBJECT_MODEL, "Say ready.", num_predict=4)
    for label, imgs in (("text only", ()), ("one 896px image", (frame,))):
        out = llm.complete(config.SUBJECT_MODEL, "Say ready.", imgs, num_predict=4)
        console.print(f"  warm {label:16s} {out.latency_s:5.2f}s   {out.prompt_tokens} prompt tokens")
    for proc in llm.client().ps().models:
        console.print(f"  resident {proc.model}: {(proc.size_vram or 0) / 2**30:.2f} GB on GPU, "
                      f"context {proc.context_length}")


def cmd_build(args) -> None:
    """Lab PDFs, the open-i index, the probe set, and every case folder."""
    from src import casebuild, labgen, probes

    for name, state in casebuild.fetch_demo_assets().items():
        console.print(f"  demo asset {name:26s} {state}", style="red" if "FAILED" in state else "")
    truth = labgen.build()
    for case_id, report in truth.items():
        casebuild.save(case_id, files={"labs": (config.CASES_DIR / case_id / "labs.pdf", "lab", "support")},
                       reference={"tests": report["tests"], "patient_id": report["patient_id"]})
    console.print(f"{len(truth)} lab reports minted")
    pair = casebuild.build_pair_case()
    console.print(f"longitudinal pair: {pair}" if pair
                  else "[yellow]no longitudinal pair - demo films not downloaded[/]")
    if config.PROBES_PATH.exists() and not args.rebuild_probes:
        gap = probes.missing_images()
        if gap:
            console.print(f"probe file present, {len(gap)} images missing - re-fetching from the Hub")
            console.print(f"re-materialised {probes.rematerialise()} studies")
        else:
            console.print(f"probes already at {config.PROBES_PATH} (pass --rebuild-probes to redo)")
        return
    index = probes.fetch_index()
    picked, counts = probes.author(index)
    scored = probes.materialise(picked, index)
    shadows = probes.shadow(scored, index)
    probes.write(scored, shadows, counts)
    console.print(f"{len(scored)} scored probes over {len({p['finding'] for p in scored})} findings, "
                  f"{len(shadows)} shadow probes -> {config.PROBES_PATH}")
    if not args.keep_orphans:
        gone, freed = probes.prune()
        console.print(f"pruned {gone} case folders no probe refers to ({freed:.0f} MB)")
    console.print("next: `probes --review`, delete anything mislabelled, then `probes --approve`")


def cmd_probes(args) -> None:
    """The hand check: review the table, drop the bad ones, stamp the rest."""
    from src import probes

    if args.approve or args.drop:
        payload = probes.approve([p for p in args.drop.split(",") if p])
        console.print(f"{len(payload['probes'])} probes stamped")
        return
    table = Table(title="probes - the checker may delete, never flip")
    for col in ("pid", "finding", "ans", "source", "sentence", "MeSH"):
        table.add_column(col)
    for p in probes.load()["probes"]:
        table.add_row(p["pid"], p["finding"], p["answer"], p["source"], p["sentence"][:110], p["mesh"][:40],
                      style="green" if p["answer"] == "yes" else "")
    console.print(table)


def cmd_gates(args) -> None:
    from src import gates

    checks = gates.run_all()
    for name, ok, detail in checks:
        console.print(f"[{'green' if ok else 'red'}]{'ok ' if ok else 'NO '}[/] {name:62s} {detail}")
    if gates.failures(checks):
        raise SystemExit(1)


def cmd_note(args) -> None:
    """The product with the audit switched off."""
    from src import casebuild, note

    case = casebuild.load(args.case)
    png = casebuild.evidence_bytes(note.current_image(case.view))
    text, out = note.write(args.model, case.view, png)
    console.print(f"[dim]{out.latency_s:.1f}s, {out.prompt_tokens} prompt tokens[/]\n{text}")


def cmd_case(args) -> None:
    """Note, claims, four probes each, verdicts - the whole trail for one case."""
    from src import casebuild, workstation

    case = casebuild.load(args.case)
    started = time.perf_counter()
    if any(e.modality == "lab" for e in case.view.evidence) and not any(e.modality == "cxr" for e in case.view.evidence):
        record = workstation.read_lab(args.model, case)
        sighted, blind = record["sighted"], record["blind"]
        table = Table(title=f"{case.view.case_id} - lab page to structured JSON")
        for col in ("condition", "image bytes", "rows", "values under their own name", "field accuracy"):
            table.add_column(col, justify="left" if col == "condition" else "right")
        accuracy = f"{sighted['score']['field_accuracy']:.0%}" if sighted["score"] else "-"
        table.add_row("sighted", f"{sighted['image_bytes_sent']:,}", str(len(sighted["tests"])),
                      str(sighted["values_recovered"]), accuracy, style="green")
        table.add_row("blind", str(blind["image_bytes_sent"]), str(len(blind["tests"])),
                      f"[bold]{blind['values_recovered']}[/]", "-",
                      style="green" if blind["values_recovered"] == 0 else "red")
        console.print(table)
        console.print(f"The blind row must read 0. It writes a plausible-looking report from nothing, and "
                      f"{blind['prose_collisions']} minted value(s) happen to appear loose in that prose - "
                      f"coincidences, not evidence, which is why the gate reads the name-value pair.")
        return
    record = workstation.read_case(args.model, case)
    console.print(f"\n[bold]{case.view.case_id}[/]  {record['manifest']['model']}  "
                  f"{time.perf_counter() - started:.0f}s\n")
    console.print(record["note"]["text"])
    table = Table(title="claim audit")
    for col in ("claim", "finding", "note", "sighted", "blind", "swapped", "repeat", "verdict", "ref"):
        table.add_column(col)
    for a in record["audits"]:
        answers = {p["condition"]: p["polarity"] for p in a["probes"]}
        style = {"prior": "yellow", "grounded": "green", "image_shaped": "magenta"}.get(a["verdict"], "red")
        table.add_row(a["cid"], a["finding"], a["asserted"], answers.get("sighted", "-"), answers.get("blind", "-"),
                      answers.get("shuffled", "-"), answers.get("repeat", "-"),
                      f"[{style}]{a['verdict']}[/]", str(a["reference"]))
    console.print(table)
    console.print(f"{record['summary']}\nnote -> {record['note_path']}")


def cmd_locate(args) -> None:
    from src import casebuild, workstation

    got = workstation.locate_case(args.model, casebuild.load(args.case), args.finding)
    console.print(got)


def cmd_compare(args) -> None:
    from src import casebuild, workstation

    got = workstation.compare_case(args.model, casebuild.load(args.case))
    console.print(f"with the real prior:      {got['real']}")
    console.print(f"with a stranger's prior:  {got['swapped']}")
    console.print(f"[bold]changes that survived swapping the prior: {got['survivors'] or 'none'}[/]")


def cmd_ct(args) -> None:
    from src import multi

    path = config.DEMO_DIR / "CT_Abdo.nii.gz" if args.volume == "CT_Abdo" else config.ROOT_DIR / args.volume
    text, answer, out = multi.ct_read(args.model, path, args.history, args.label, args.slices)
    console.print(f"[dim]{args.slices} slices, {out.prompt_tokens} prompt tokens, {out.latency_s:.1f}s "
                  f"- demonstrated, not measured[/]\n{text}\n\n[bold]Final answer: {answer}[/]")


def cmd_ladder(args) -> None:
    from src import audit

    table = Table(title="the four conditions - one switch each")
    for col in ("condition", "images", "seed", "isolates", "prediction, written before the run"):
        table.add_column(col)
    for arm in audit.LADDER:
        table.add_row(arm.name, arm.images, str(arm.seed), arm.isolates, arm.prediction)
    console.print(table)


def cmd_run(args) -> None:
    from src import harness

    harness.run(args.profile, resume=not args.restart)


def cmd_report(args) -> None:
    from src import report

    report.render()
    console.print(f"\nwrote {config.LEADERBOARD_PATH}")


def cmd_preflight(args) -> None:
    from src import preflight

    for which in args.which or ["merge", "canary", "vram"]:
        getattr(preflight, {"merge": "image_merge", "boxes": "boxes_q4_vs_q8"}.get(which, which))()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="second-read",
        description="A local case workstation on MedGemma 1.5. Every claim in a generated note is "
                    "re-asked with the image deleted and with another patient's image in its place. "
                    "Start with `case`: it shows for one film every number the leaderboard aggregates.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("models", help="tags, digests, warm timings").set_defaults(fn=cmd_models)
    build = sub.add_parser("build", help="demo assets, lab PDFs, probes, case folders")
    build.add_argument("--rebuild-probes", action="store_true",
                       help="re-author the probe set from scratch; needs a fresh hand check afterwards")
    build.add_argument("--keep-orphans", action="store_true",
                       help="do not delete case folders the new probe set no longer refers to")
    build.set_defaults(fn=cmd_build)
    pr = sub.add_parser("probes", help="review or approve the probe set")
    pr.add_argument("--drop", default="", help="comma-separated pids to delete")
    pr.add_argument("--approve", action="store_true")
    pr.set_defaults(fn=cmd_probes)
    sub.add_parser("gates", help="every free validity check").set_defaults(fn=cmd_gates)
    per_case = (
        ("note", cmd_note, (), "the note alone, with the audit switched off"),
        ("case", cmd_case, (), "the note plus the full audit trail - run this one first"),
        ("compare", cmd_compare, (), "prior vs current, then the same with a stranger's prior"),
        ("locate", cmd_locate, ("--finding",), "draw the box, and test whether it moves patient to patient"),
    )
    for name, fn, extra, blurb in per_case:
        p = sub.add_parser(name, help=blurb)
        p.add_argument("case", help="a case id like CXR-0272, LAB-01 or PAIR-demo, or a folder path")
        p.add_argument("--model", default=config.SUBJECT_MODEL)
        for flag in extra:
            p.add_argument(flag, default="cardiomegaly", help="a finding from the vocabulary in claims.py")
        p.set_defaults(fn=fn)
    ct = sub.add_parser("ct", help="read a CT volume as a slice stack (demonstrated, not measured)")
    ct.add_argument("volume", nargs="?", default="CT_Abdo")
    ct.add_argument("--slices", type=int, default=config.CT_SLICES_DEFAULT)
    ct.add_argument("--history", default="abdominal pain")
    ct.add_argument("--label", default="free intraperitoneal air")
    ct.add_argument("--model", default=config.SUBJECT_MODEL)
    ct.set_defaults(fn=cmd_ct)
    sub.add_parser("ladder", help="the conditions and their predictions").set_defaults(fn=cmd_ladder)
    run = sub.add_parser("run", help="probes x conditions x models")
    run.add_argument("--profile", default="lite", choices=("smoke", "lite", "full"))
    run.add_argument("--restart", action="store_true")
    run.set_defaults(fn=cmd_run)
    sub.add_parser("report", help="gates, tables, verdict, leaderboard").set_defaults(fn=cmd_report)
    pf = sub.add_parser("preflight", help="merge | canary | vram | boxes")
    pf.add_argument("which", nargs="*")
    pf.set_defaults(fn=cmd_preflight)
    return parser


if __name__ == "__main__":
    _args = build_parser().parse_args()
    _args.fn(_args)
