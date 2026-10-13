"""Describe a built dataset, and say whether there is enough of it.

Inputs:  Pairs, and the reports the earlier stages produced
Outputs: tables on the terminal

The honest numbers for a style LoRA, which are far lower than the ones people
quote for teaching a model facts:

    under 200 pairs   it parrots whole lines back
    500 to 1000       the voice genuinely carries
    2000 and up       comfortable

A verdict line, not a grade. The reader is the one who knows whether the chat
they picked is representative of how they talk.
"""

from __future__ import annotations

from statistics import median

from rich.console import Console
from rich.table import Table

from src.clean import CleanReport, FilterReport
from src.config import SHORT_REPLY_CHARS
from src.parse import ParseReport
from src.sessionize import Pair

console = Console()


def _percentile(values: list[int], fraction: float) -> int:
    """The value below which the given fraction of the sample falls."""
    if not values:
        return 0
    ordered = sorted(values)
    index = min(int(fraction * len(ordered)), len(ordered) - 1)
    return ordered[index]


def verdict(pair_count: int) -> tuple[str, str]:
    """Turn a pair count into a sentence and a colour."""
    if pair_count < 200:
        return ("Expect parroting. The model will repeat whole lines from this "
                "chat rather than learn its shape. Add another export.", "red")
    if pair_count < 500:
        return ("Enough to see the voice move, not enough to be convincing. "
                "Worth adding a second chat if you have one.", "yellow")
    if pair_count < 2000:
        return ("A good size for a style LoRA. The voice should carry.", "green")
    return ("Comfortable. You could hold back more for testing.", "green")


def show_parse(report: ParseReport) -> None:
    """Print what the parser found, including who is in the file."""
    table = Table(title="the export", title_justify="left", show_header=False)
    table.add_row("messages", f"{report.messages:,}")
    table.add_row("notices skipped", f"{report.system_lines:,}")
    table.add_row("continuation lines", f"{report.continuation_lines:,}")
    table.add_row("date format", report.date_order)
    console.print(table)

    senders = Table(title="who is in this chat", title_justify="left")
    senders.add_column("sender")
    senders.add_column("messages", justify="right")
    for name, count in report.senders.items():
        senders.add_row(name, f"{count:,}")
    console.print(senders)
    console.print("[dim]If the date format above is wrong, the sessions will "
                  "be cut in the wrong places.[/dim]\n")


def show_build(clean: CleanReport, filtered: FilterReport,
               pairs: list[Pair]) -> None:
    """Print what cleaning removed and what the dataset now looks like."""
    dropped = Table(title="what came out", title_justify="left", show_header=False)
    dropped.add_row("messages kept", f"{clean.kept:,}")
    dropped.add_row("no words in them", f"{clean.noise_dropped:,}")
    dropped.add_row("links masked", f"{clean.links:,}")
    for name, count in sorted(clean.redactions.items()):
        dropped.add_row(f"{name}s redacted", f"{count:,}")
    dropped.add_row("replies too long", f"{filtered.too_long:,}")
    dropped.add_row("duplicate replies", f"{filtered.duplicates:,}")
    dropped.add_row("one-worders trimmed", f"{filtered.short_downsampled:,}")
    console.print(dropped)

    if not pairs:
        console.print("[red]No pairs. Check that --me matches a sender "
                      "exactly.[/red]")
        return

    replies = [len(p.reply) for p in pairs]
    contexts = [sum(len(t.text) for t in p.context) for p in pairs]
    words = {w for p in pairs for w in p.reply.casefold().split()}
    short = sum(1 for r in replies if r <= SHORT_REPLY_CHARS)

    shape = Table(title="the dataset", title_justify="left", show_header=False)
    shape.add_row("training pairs", f"{len(pairs):,}")
    shape.add_row("reply length", f"median {median(replies):.0f}, "
                                  f"p90 {_percentile(replies, 0.9)}, "
                                  f"longest {max(replies)} chars")
    shape.add_row("context length", f"median {median(contexts):.0f} chars")
    shape.add_row("one-word replies", f"{short / len(pairs):.0%}")
    shape.add_row("distinct words", f"{len(words):,}")
    console.print(shape)

    sentence, colour = verdict(len(pairs))
    console.print(f"[{colour}]{sentence}[/{colour}]\n")


def show_examples(pairs: list[Pair], count: int = 3) -> None:
    """Print real pairs. Counting rows proves nothing; reading them does."""
    console.print("[bold]a few pairs, as the model will see them[/bold]")
    step = max(1, len(pairs) // count)
    for pair in pairs[::step][:count]:
        console.print(f"[dim]{pair.pair_id}[/dim]")
        for turn in pair.as_chat()["messages"]:
            who = "you " if turn["role"] == "assistant" else "them"
            body = turn["content"].replace("\n", " / ")
            console.print(f"  [cyan]{who}[/cyan]  {body}")
        console.print()


if __name__ == "__main__":
    checks = [
        ("under 200 warns", verdict(150)[1] == "red"),
        ("500 is workable", verdict(600)[1] == "green"),
        ("median of an even list", _percentile([1, 2, 3, 4], 0.5) == 3),
        ("p90 of a hundred", _percentile(list(range(100)), 0.9) == 90),
        ("empty list is safe", _percentile([], 0.5) == 0),
    ]
    for name, ok in checks:
        print(f"{'ok  ' if ok else 'FAIL'}  {name}")
    raise SystemExit(0 if all(ok for _, ok in checks) else 1)
