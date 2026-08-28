"""The measurement that decides whether the rest of the project is affordable.

Inputs:  a Groq API key
Outputs: cache behaviour, tokenizer calibration, completion length, truncation

Everything downstream rests on one claim from Groq's docs: cached tokens do not
count toward the rate limit. If it holds, a second probe against the same
prefix costs ~130 tokens and the full ablation fits in a day. If it does not,
every probe costs a full prefix, the run is roughly 4x over the daily ceiling,
and the pre-registered fallback fires.

`cached_tokens` alone cannot test that claim - it reports the PRICE discount,
not the rate-limit bucket. So this reads x-ratelimit-remaining-tokens around
each call and compares the observed drain against the two hypotheses.

It also calibrates the tokenizer. tiktoken prices raw text; the Harmony chat
template adds framing it never sees. Every budget assertion in the project is
only as honest as the ratio measured here.
"""

from __future__ import annotations

import asyncio
import time

from rich.console import Console
from rich.table import Table

from src import cache, config, llm, tokens

console = Console()

_FILLER = (
    "PaySetu settlement note: batch {i} reconciled against the merchant ledger "
    "with no variance; payout rail nominal; no operator action required. "
)

QUESTIONS = [
    "In one word, what rail is described above?",
    "In one word, was any variance found?",
    "In one word, is operator action required?",
    "In one word, what was reconciled above?",
    "In one word, what state is the payout rail in?",
]


def _prefix(target_tokens: int) -> str:
    """Build a deterministic prefix of roughly `target_tokens` tokens."""
    parts: list[str] = []
    i = 0
    while tokens.count_text("".join(parts)) < target_tokens:
        parts.append(_FILLER.format(i=i))
        i += 1
    return "".join(parts)


def _remaining(headers: dict[str, str]) -> int | None:
    """Per-minute token headroom the server reports, if it reports one."""
    raw = headers.get("x-ratelimit-remaining-tokens")
    try:
        return int(raw) if raw is not None else None
    except ValueError:
        return None


async def run(target_tokens: int = 0) -> dict[str, float]:
    """Fire calls against one shared prefix, fast, and report what they cost.

    The prefix is deliberately SMALLER than the project's context budget. At
    3,000 tokens only two calls fit the 8,000-token bucket before a 429, and
    the retry backoff then stretches the run long enough for refill to swamp
    the measurement - the bucket refills at TOKENS_PER_MINUTE/60 per second, so
    a slow test measures refill rather than spend. A ~1,200-token prefix lets
    six calls run back to back, which is what makes the two hypotheses
    separable.
    """
    target = target_tokens or 1_200
    prefix = _prefix(target)
    local_prefix_tokens = tokens.count_text(prefix)

    console.print(
        f"[bold]cache pre-flight[/bold]  model={config.CHAT_MODEL}  "
        f"prefix={local_prefix_tokens:,} tok (local count)  "
        f"reserve={config.MAX_COMPLETION_TOKENS}  "
        f"refill={config.TOKENS_PER_MINUTE / 60:.0f} tok/s"
    )

    async def ask(question: str) -> tuple[object, float]:
        """One un-spaced, un-cached call. Returns the result and its timestamp."""
        messages = [
            {"role": "system", "content": prefix},
            {"role": "user", "content": question},
        ]
        result = await llm.complete(messages, use_cache=False)
        return result, time.perf_counter()

    # Warm-up. The cache did not engage until the fourth identical prefix on a
    # first run, so measuring from call one would report a miss that is really
    # a cold start.
    console.print("warming the prefix ...")
    for _ in range(2):
        await ask(QUESTIONS[0])

    table = Table(show_header=True, header_style="bold")
    for col in ("call", "prompt", "cached", "compl", "finish",
                "tpm left", "drain", "refill", "net"):
        table.add_column(col, justify="right")

    rows: list[dict[str, float | None]] = []
    previous_remaining: int | None = None
    previous_at: float | None = None
    started = time.perf_counter()
    rate_limited = 0

    for n, question in enumerate(QUESTIONS + QUESTIONS, start=1):
        result, at = await ask(question)
        if result.finish_reason == "rate_limited":
            rate_limited += 1
        remaining = _remaining(dict(result.headers))
        drain = refill = net = None
        if previous_remaining is not None and remaining is not None:
            drain = previous_remaining - remaining
            refill = (at - previous_at) * config.TOKENS_PER_MINUTE / 60.0
            # The bucket both drains and refills between two observations, so
            # what the call actually consumed is the observed drop plus
            # whatever refilled while it ran.
            net = drain + refill
        rows.append({
            "prompt": result.usage.prompt_tokens,
            "cached": result.usage.cached_tokens,
            "completion": result.usage.completion_tokens,
            "net": net,
        })
        table.add_row(
            str(n),
            f"{result.usage.prompt_tokens:,}",
            "n/a" if result.usage.cached_tokens is None
            else f"{result.usage.cached_tokens:,}",
            str(result.usage.completion_tokens),
            result.finish_reason,
            "?" if remaining is None else f"{remaining:,}",
            "-" if drain is None else f"{drain:,}",
            "-" if refill is None else f"{refill:,.0f}",
            "-" if net is None else f"{net:,.0f}",
        )
        previous_remaining, previous_at = remaining, at

    elapsed = time.perf_counter() - started
    console.print(table)

    priced = [r for r in rows if r["prompt"]]
    hits = [r for r in priced if r["cached"]]
    hit_rate = len(hits) / len(priced) if priced else 0.0
    cached_frac = (
        sum(float(r["cached"] or 0) for r in priced)
        / sum(float(r["prompt"]) for r in priced)
        if priced else 0.0
    )
    mean_completion = sum(float(r["completion"]) for r in rows) / max(len(rows), 1)
    mean_prompt = sum(float(r["prompt"]) for r in priced) / max(len(priced), 1)
    fudge = mean_prompt / max(local_prefix_tokens, 1)

    nets = [float(r["net"]) for r in rows if r["net"] is not None and r["cached"]]
    mean_net = sum(nets) / len(nets) if nets else 0.0
    if_exempt = mean_completion
    if_charged = mean_prompt + mean_completion
    midpoint = (if_exempt + if_charged) / 2
    decided = bool(nets)
    exempt = decided and mean_net < midpoint

    console.print()
    console.print(f"calls with a cache hit       : [bold]{hit_rate:.0%}[/bold] "
                  f"({len(hits)}/{len(priced)}), {cached_frac:.0%} of prompt tokens")
    console.print(f"429s during the probe        : {rate_limited}")
    if decided:
        console.print(f"refill-corrected spend/call  : {mean_net:,.0f} tok "
                      f"(exempt ~{if_exempt:,.0f}, charged ~{if_charged:,.0f})")
        console.print(f"cached tokens rate-limit exempt: "
                      f"[bold]{'YES' if exempt else 'NO'}[/bold]")
    else:
        console.print("[yellow]no cache hit landed - cannot decide exemption[/yellow]")
    console.print(f"tokenizer fudge              : [bold]{fudge:.3f}[/bold] "
                  f"(groq {mean_prompt:,.0f} / local {local_prefix_tokens:,})")
    console.print(f"mean completion              : {mean_completion:,.0f} tok "
                  f"of a {config.MAX_COMPLETION_TOKENS} reserve")
    truncated = sum(1 for r in rows if r["completion"] >= config.MAX_COMPLETION_TOKENS)
    console.print(f"truncation rate              : {truncated / len(rows):.0%} "
                  f"(ceiling {config.TRUNCATION_CEILING:.0%})")
    console.print(f"wall clock                   : {elapsed:.1f}s for {len(rows)} calls")

    spent, requests = cache.spent_today()
    console.print(f"ledger today                 : {spent:,} tok / {requests} requests "
                  f"of {config.TOKENS_PER_DAY:,} / {config.REQUESTS_PER_DAY}")

    console.print()
    if not (decided and exempt and cached_frac >= 0.80):
        console.print(
            "[bold yellow]Pre-registered fallback fires.[/bold yellow] Run "
            "--profile lite (none / raw / window / cap+pin+window / full / no-pin), "
            "one model, and report the reduced n as reduced."
        )
    else:
        console.print("[bold green]Full profile is affordable.[/bold green]")
    console.print(f"Set TOKENIZER_FUDGE={fudge:.3f} in .env before the run.")

    return {
        "hit_rate": hit_rate,
        "cached_frac": cached_frac,
        "fudge": fudge,
        "mean_completion": mean_completion,
        "mean_net": mean_net,
        "exempt": float(exempt),
    }


if __name__ == "__main__":
    asyncio.run(run())
