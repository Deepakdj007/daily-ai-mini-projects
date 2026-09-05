"""The control that asks whose defence it actually is: yours, or the model's.

Inputs:  a list of chat model ids
Outputs: attack leak counts for prompt-only and full-stack, per model

A system prompt full of security rules is not a control you own. Its strength
is a property of whichever model is reading it. Swap the model for a cheaper
one and the rules do not change - the protection does. The deterministic
layers are the only part of the stack that behaves the same either way.
"""

from __future__ import annotations

import asyncio

from src.evaluate import CONCURRENCY, _run_attack
from src.ladder import CONFIGS_BY_NAME
from src.suites import ATTACKS

# Three sizes on the same free tier, same prompt, same attacks.
SWAP_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
]

SWAP_CONFIGS = ["prompt-only", "full stack"]


async def run_swap(models: list[str] | None = None) -> dict:
    """Run the attack suite against each model under two configurations."""
    models = models or SWAP_MODELS
    out: dict[str, dict] = {}

    for model in models:
        out[model] = {}
        for cfg_name in SWAP_CONFIGS:
            cfg = CONFIGS_BY_NAME[cfg_name]
            gate = asyncio.Semaphore(CONCURRENCY)

            async def one(case):
                """Run one attack under the concurrency gate."""
                async with gate:
                    return await _run_attack(case, cfg, model)

            outcomes = await asyncio.gather(*(one(c) for c in ATTACKS))
            leaked = [o.case_id for o in outcomes if o.status == "leaked"]
            out[model][cfg_name] = {
                "attacks": len(outcomes),
                "leaked": len(leaked),
                "leaked_ids": leaked,
            }
            print(
                f"  {model:26} {cfg_name:13} leaked "
                f"{len(leaked)}/{len(outcomes)}",
                flush=True,
            )
    return out
