"""The command line.

Inputs:  a WhatsApp export
Outputs: out/train.jsonl, out/heldout.jsonl, out/stranger.jsonl

    inspect   who is in this export, and did the dates parse
    build     turn it into a training set
    stats     describe a set that was already built
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.table import Table

from src.clean import clean_messages, filter_pairs
from src.config import (FACTS_PATH, GATED_FACTS_PATH, HELDOUT_PAIRS,
                        HELDOUT_PATH, OUT_DIR, RESPONSES_PATH, SAMPLE_CHAT,
                        STRANGER_PATH, TRAIN_PATH, ensure_dirs, resolve_me)
from src.facts import (MIN_USABLE_FACTS, check_facts, load_facts,
                       scan_candidates, usable)
from src.parse import parse_export
from src.report import console, show_build, show_examples, show_parse
from src.score import (conclude, grade_facts, load_responses, show_facts,
                       show_misses, show_style)
from src.sessionize import build_pairs, is_group_chat, split_heldout


def _write(path: Path, pairs: list) -> None:
    """Write pairs as JSON lines, the format every trainer reads."""
    with path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair.as_chat(), ensure_ascii=False) + "\n")


def cmd_inspect(args: argparse.Namespace) -> None:
    """Show what is in the export before committing to anything."""
    _, report = parse_export(args.chat)
    show_parse(report)
    console.print("Pick your own name from that list, then run:\n"
                  f"  [cyan]PYTHONPATH=. uv run python -m src.main build "
                  f"--me \"{next(iter(report.senders), 'Your Name')}\"[/cyan]")


def cmd_build(args: argparse.Namespace) -> None:
    """Export -> training pairs, held-out pairs, and the control set."""
    ensure_dirs()
    messages, parsed = parse_export(args.chat)
    show_parse(parsed)

    me = resolve_me(list(parsed.senders), args.me)
    others = [name for name in parsed.senders if name != me]
    if not others:
        raise SystemExit("This export has only one sender, so there is no "
                         "conversation to learn from.")
    stranger = others[0]
    group = is_group_chat(parsed.senders)

    cleaned, clean_report = clean_messages(messages)
    pairs, filter_report = filter_pairs(build_pairs(cleaned, me, group))
    show_build(clean_report, filter_report, pairs)
    if not pairs:
        return
    show_examples(pairs)

    train, held = split_heldout(pairs, HELDOUT_PAIRS)
    # The same treatment for someone else in the chat. If the fine-tune scores
    # their replies better too, it learned what WhatsApp looks like, not you.
    control, _ = filter_pairs(build_pairs(cleaned, stranger, group, prefix="c"))
    _, control_held = split_heldout(control, HELDOUT_PAIRS)

    _write(TRAIN_PATH, train)
    _write(HELDOUT_PATH, held)
    _write(STRANGER_PATH, control_held)

    console.print(f"[green]{len(train):,}[/green] training pairs  "
                  f"[dim]{TRAIN_PATH.relative_to(OUT_DIR.parent)}[/dim]")
    console.print(f"[green]{len(held):,}[/green] held out for scoring  "
                  f"[dim]{HELDOUT_PATH.relative_to(OUT_DIR.parent)}[/dim]")
    console.print(f"[green]{len(control_held):,}[/green] held out for "
                  f"{stranger}, the control  "
                  f"[dim]{STRANGER_PATH.relative_to(OUT_DIR.parent)}[/dim]")


def _rebuild(args: argparse.Namespace) -> tuple[list, list]:
    """Recreate the train / held-out split exactly as `build` made it."""
    messages, parsed = parse_export(args.chat)
    me = resolve_me(list(parsed.senders), args.me)
    cleaned, _ = clean_messages(messages)
    pairs, _ = filter_pairs(build_pairs(cleaned, me, is_group_chat(parsed.senders)))
    return split_heldout(pairs, HELDOUT_PAIRS)


def cmd_facts(args: argparse.Namespace) -> None:
    """Find answer-shaped things that were said, or gate a written probe set."""
    train, _ = _rebuild(args)
    if args.action == "scan":
        console.print("[bold]said in the training split, rarest first[/bold]")
        console.print("[dim]Write a question for the ones you can answer. Ask "
                      "it in your own words, not the chat's.[/dim]\n")
        table = Table()
        table.add_column("value")
        table.add_column("times said", justify="right")
        table.add_column("first seen in")
        for span, count, ids in scan_candidates(train):
            table.add_row(span, str(count), ", ".join(ids))
        console.print(table)
        return

    facts = check_facts(load_facts(FACTS_PATH), train)
    table = Table(title=f"{FACTS_PATH.name} against {len(train)} training pairs",
                  title_justify="left")
    table.add_column("id")
    table.add_column("answer")
    table.add_column("said", justify="right")
    table.add_column("verdict")
    for fact in facts:
        colour = "green" if fact.status == "usable" else "red"
        table.add_row(fact.fact_id, fact.gold, str(fact.train_occurrences),
                      f"[{colour}]{fact.status}[/{colour}]"
                      + (f" - {fact.reason}" if fact.reason else ""))
    console.print(table)

    kept = usable(facts)
    console.print(f"[bold]{len(kept)}[/bold] of {len(facts)} questions are usable.")
    if len(kept) < MIN_USABLE_FACTS:
        console.print(f"[red]Below {MIN_USABLE_FACTS}, a percentage would be "
                      f"meaningless. Report counts instead.[/red]")

    # Write the survivors separately. This is the file that goes to Kaggle, so
    # the rejected questions are never asked in the first place.
    ensure_dirs()
    GATED_FACTS_PATH.write_text(json.dumps(
        [{"id": f.fact_id, "question": f.question, "gold": f.gold,
          "aliases": f.aliases} for f in kept], ensure_ascii=False, indent=2),
        encoding="utf-8")
    console.print(f"[dim]{GATED_FACTS_PATH} written - upload this one, not "
                  f"data/facts.json[/dim]")


def _graded(args: argparse.Namespace) -> tuple[dict, list[dict]]:
    """Load the notebook's output and grade its fact answers."""
    data = load_responses(args.responses)
    if data["run"].get("synthetic"):
        console.print("[red on white] SYNTHETIC FIXTURE - these numbers are "
                      "invented so the scorer can run without a GPU. They are "
                      "not a training run. [/red on white]\n")
    train, _ = _rebuild(args)
    facts = check_facts(load_facts(FACTS_PATH), train)
    return data, grade_facts(facts, data["facts"])


def cmd_score(args: argparse.Namespace) -> None:
    """The whole point: did the voice move, and did the facts come with it."""
    data, graded = _graded(args)
    you_gain, her_gain, moved = show_style(data)
    show_facts(graded)
    conclude(you_gain, her_gain, moved, graded)


def cmd_misses(args: argparse.Namespace) -> None:
    """Read every miss. The grader cannot see a paraphrase and you can."""
    _, graded = _graded(args)
    show_misses(graded)


def cmd_stats(args: argparse.Namespace) -> None:
    """Describe a dataset that was already built."""
    if not TRAIN_PATH.exists():
        raise SystemExit(f"{TRAIN_PATH} does not exist yet. Run `build` first.")
    rows = [json.loads(line) for line in
            TRAIN_PATH.read_text(encoding="utf-8").splitlines()]
    replies = [r["messages"][-1]["content"] for r in rows]
    console.print(f"{len(rows):,} pairs in {TRAIN_PATH.name}")
    console.print(f"median reply {sorted(len(r) for r in replies)[len(replies) // 2]} chars")
    for reply in replies[:5]:
        console.print(f"  [cyan]you[/cyan]  {reply[:100]}")


def main() -> None:
    """Parse the command line and run the chosen command."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--chat", type=Path, default=SAMPLE_CHAT,
                        help="the exported .txt (default: the sample chat)")
    common.add_argument("--me", default=None,
                        help="which sender is you, exactly as it appears")
    common.add_argument("--responses", type=Path, default=RESPONSES_PATH,
                        help="the notebook's responses.json")

    parser = argparse.ArgumentParser(prog="understudy", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, handler, help_text in (
        ("inspect", cmd_inspect, "who is in this export"),
        ("build", cmd_build, "turn it into a training set"),
        ("stats", cmd_stats, "describe a built set"),
        ("score", cmd_score, "did the voice move, did the facts follow"),
        ("misses", cmd_misses, "read every fact the tuned model missed"),
    ):
        sub.add_parser(name, help=help_text, parents=[common]
                       ).set_defaults(handler=handler)

    facts = sub.add_parser("facts", help="find or gate fact questions",
                           parents=[common])
    facts.add_argument("action", choices=["scan", "check"])
    facts.set_defaults(handler=cmd_facts)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
