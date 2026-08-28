"""The live demo: the same session with the layer off, then on.

Inputs:  a turn count, or a single question and a depth
Outputs: a per-turn ledger of what each design cost, and where one of them stops

Two commands.

`ask` is the one to run first. It puts a question about something said near the
start of a long session to the unlayered agent and then to the layered one, and
prints both answers. It is one comparison, it costs almost nothing, and it is
the whole project in ten seconds.

`demo` walks a session forward turn by turn, printing assembled tokens, what
the provider reported as cached, and the running spend. It is where the break
actually happens rather than being asserted: the unlayered design keeps
growing until a request is larger than the per-minute bucket, and then it is
rejected outright rather than queued.
"""

from __future__ import annotations

import asyncio

from rich.console import Console
from rich.table import Table

from src import config, llm, summarizer
from src.ladder import BY_NAME
from src.pipeline import assemble, make_deps
from src.scenario import build

console = Console()


def _deps():
    """Layer dependencies wired to the frozen summary."""
    fixture = summarizer.load() or {"text": ""}
    return make_deps(summarize_fn=summarizer.as_fn(fixture), fudge=config.TOKENIZER_FUDGE)


async def ask(question: str, turn: int = 0) -> None:
    """Put one question to the unlayered agent and then to the layered one."""
    turns, _, spec = build()
    depth = turn or len(turns)
    history = turns[:depth]
    deps = _deps()

    console.print(f"[bold]{question}[/bold]")
    console.print(f"asked {depth} turns into the session\n")

    for arm_name in ("raw", "full"):
        ctx = await assemble(
            history, question, BY_NAME[arm_name].config, deps,
            system_prompt=spec["system_prompt"], pins=spec["pinned_facts"],
            instruction=config.ANSWER_INSTRUCTION,
        )
        result = await llm.complete(
            ctx.messages(),
            cache_key=f"ask-{arm_name}-{depth}-{hash(question) & 0xffffff}",
            local_prompt_tokens=ctx.used(),
        )
        label = "no layer" if arm_name == "raw" else "with the layer"
        body = result.text.strip() or f"[red]{result.error}[/red]"
        console.print(
            f"[bold]{label}[/bold]  ({ctx.used():,} tok assembled, "
            f"{ctx.turns_in_context()} turns held)"
        )
        console.print(f"  {body}\n")


async def demo(max_turns: int = 12) -> None:
    """Walk the session forward, printing what each design costs per turn."""
    turns, _, spec = build()
    deps = _deps()
    question = "What is the current state of the incident?"
    totals: dict[str, int] = {}

    for arm_name in ("raw", "full"):
        label = "no layer - raw history" if arm_name == "raw" else "with the layer"
        table = Table(title=f"{label}")
        for col in ("turn", "assembled", "cached", "billable", "status"):
            table.add_column(col, justify="right")

        billable_total = 0
        broke_at: int | None = None
        for n in range(2, max_turns + 1):
            ctx = await assemble(
                turns[:n], question, BY_NAME[arm_name].config, deps,
                system_prompt=spec["system_prompt"], pins=spec["pinned_facts"],
                instruction=config.ANSWER_INSTRUCTION,
            )
            result = await llm.complete(
                ctx.messages(),
                cache_key=f"demo-{arm_name}-{n}",
                local_prompt_tokens=ctx.used(),
            )
            cached = result.usage.cached_tokens
            billable = result.usage.uncached_tokens
            billable_total += billable
            if result.finish_reason != "stop" and broke_at is None:
                broke_at = n
            table.add_row(
                str(n), f"{ctx.used():,}",
                "n/a" if cached is None else f"{cached:,}",
                f"{billable:,}",
                result.finish_reason,
            )
            if broke_at is not None:
                break

        console.print(table)
        console.print(f"  billable over the session: [bold]{billable_total:,}[/bold] tok")
        if broke_at:
            console.print(
                f"  [bold red]stopped at turn {broke_at}[/bold red] - the request "
                f"grew past the {config.TOKENS_PER_MINUTE:,}-token bucket and was "
                f"rejected, not queued\n"
            )
        else:
            console.print(f"  completed all {max_turns} turns under budget\n")
        totals[arm_name] = billable_total

    console.print(
        f"Over {max_turns} turns the layer cost [bold]{totals['full']:,}[/bold] "
        f"billable tokens against [bold]{totals['raw']:,}[/bold] for raw "
        f"history - {1 - totals['full'] / max(totals['raw'], 1):.0%} less.\n"
        f"That is the opposite of what an ideal prefix cache predicts, where "
        f"append-only history is the cheapest thing you can send. It holds here "
        f"because Groq's cache barely engaged - look at the cached column, "
        f"mostly n/a. Prefix stability only pays when the provider's cache "
        f"actually fires, and on this account it mostly did not.\n"
    )

    # The wall itself. Rejected before dispatch, so showing it costs nothing.
    far = await assemble(
        turns, question, BY_NAME["raw"].config, deps,
        system_prompt=spec["system_prompt"], pins=spec["pinned_facts"],
        instruction=config.ANSWER_INSTRUCTION,
    )
    broken = await llm.complete(
        far.messages(), local_prompt_tokens=far.used(), use_cache=False
    )
    console.print(
        f"[bold]The same agent at turn {len(turns)}, unlayered[/bold]\n"
        f"  assembled {far.used():,} tok\n"
        f"  [red]{broken.finish_reason}: {broken.error}[/red]\n"
        f"  The model's context window is 131,072 tokens. This request is "
        f"nowhere near it, and it still cannot run."
    )


if __name__ == "__main__":
    asyncio.run(demo())
