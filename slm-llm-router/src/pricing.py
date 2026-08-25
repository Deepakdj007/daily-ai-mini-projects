"""Cost accounting for every model call.

Inputs:  a model ID plus prompt/completion token counts.
Outputs: cost in USD.

The whole project is an argument about money, so this table is the most
load-bearing twenty lines in the repo. Rates verified 2026-08-15 against
console.groq.com/docs/models.
"""

from __future__ import annotations

# USD per 1M tokens, as (input_rate, output_rate).
#
# The two gpt-oss models are the entire Groq production text lineup as of
# 2026-08-16, the day llama-3.3-70b-versatile and llama-3.1-8b-instant shut
# down. Note the gap between them is only 2x - that is why a cloud-only router
# cannot reach a 90% saving no matter how clever it is.
#
# qwen/qwen3.6-27b is deliberately absent. Groq's own deprecation page
# recommends it as the llama-3.3-70b replacement, but it is preview status and
# costs $0.60/$3.00 - roughly 4x MORE than gpt-oss-120b. Following that
# recommendation would raise your bill.
PRICING: dict[str, tuple[float, float]] = {
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),
}

# Locally-run models have no per-token API price. This is marginal API cost,
# not true cost: electricity and the hardware you already bought are real, but
# they are amortised and effectively zero per query. The guide says this out
# loud rather than pretending local inference is free.
LOCAL_RATE = (0.0, 0.0)


def rates_for(model: str) -> tuple[float, float]:
    """Look up (input, output) USD-per-1M rates for a model.

    Raises with the offending model name rather than a bare KeyError, which is
    the failure mode in agent-eval-arena/src/config.py when a new model is
    added to a run but not to the price table.
    """
    if model in PRICING:
        return PRICING[model]
    if is_local(model):
        return LOCAL_RATE
    raise KeyError(
        f"No price on file for model {model!r}. "
        f"Add a row to PRICING in src/pricing.py. "
        f"Known models: {sorted(PRICING)}"
    )


def is_local(model: str) -> bool:
    """True for models served by local Ollama, which cost nothing per token."""
    return model.startswith("local:")


def calculate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Compute the real-money cost of one call from its token counts."""
    input_rate, output_rate = rates_for(model)
    return (prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000


if __name__ == "__main__":
    # A typical routed query: ~250 prompt tokens, ~60 completion tokens.
    for model in ("openai/gpt-oss-120b", "openai/gpt-oss-20b", "local:qwen3.5:4b"):
        cost = calculate_cost_usd(model, 250, 60)
        print(f"{model:24s} {cost:.8f} USD   ({cost * 1000:.5f} USD per 1k queries)")

    try:
        calculate_cost_usd("some/unpriced-model", 1, 1)
    except KeyError as exc:
        print(f"\nunpriced model raises cleanly: {exc}")
