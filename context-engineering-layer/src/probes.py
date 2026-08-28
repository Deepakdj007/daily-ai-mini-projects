"""Run one probe against one arm, and record what the context actually held.

Inputs:  a transcript, a Fact, an Arm
Outputs: a ProbeResult

The `fact_present` grep is the most important thing this file does. Recall
alone cannot tell you whether a layer failed or the model did: an arm that
scores 40% might be dropping the facts, or holding them and failing to read
them. Those need opposite fixes, and only the 2x2 of (present, correct)
separates them.

It uses the SAME matcher that grades the answer. Two different matchers here
would disagree at the margin and manufacture contamination events out of
nothing but a formatting difference.
"""

from __future__ import annotations

import json
from typing import Sequence

from src import cache, config, llm
from src.context import Context
from src.grader import classify, matches
from src.ladder import Arm
from src.pipeline import assemble
from src.layers.base import LayerDeps
from src.types import Fact, ProbeResult, Turn, Usage


def cache_key(arm: Arm, fact: Fact, model: str, scenario_hash: str,
              summary_hash: str) -> str:
    """Everything that can change an answer, hashed into one key.

    Leave any of these out and an edit elsewhere returns a stale answer that
    still looks fresh. Raw completions are stored and graded separately, so a
    grader fix costs zero requests to apply.
    """
    return cache.key(model, {
        "arm": arm.name,
        "probe": fact.id,
        "temperature": config.TEMPERATURE,
        "max_completion_tokens": config.MAX_COMPLETION_TOKENS,
        "reasoning_effort": config.REASONING_EFFORT,
        "budget": arm.config.context_budget,
        "switches": arm.switch_map(),
        "prompt_version": config.PROMPT_VERSION,
        "assembler_version": config.ASSEMBLER_VERSION,
        "tokenizer": config.TOKENIZER_NAME,
        "fudge": config.TOKENIZER_FUDGE,
        "scenario": scenario_hash,
        "summary": summary_hash,
    })


def _assert_grep_agrees(ctx: Context, fact: Fact) -> None:
    """The flattened text and the API payload must contain the same facts.

    Cheap, and it protects the single most important column: if messages()
    ever dropped a block that text() still rendered, every fact_present value
    would be quietly wrong in the direction that looks like contamination.
    """
    from_text = matches(fact.value, ctx.text())
    from_payload = matches(fact.value, json.dumps(ctx.messages(), ensure_ascii=False))
    if from_text != from_payload:
        raise AssertionError(
            f"{fact.id}: fact_present disagrees between Context.text() "
            f"({from_text}) and the rendered payload ({from_payload})"
        )


async def run_probe(
    turns: Sequence[Turn],
    fact: Fact,
    arm: Arm,
    deps: LayerDeps,
    *,
    model: str,
    system_prompt: str,
    pins: Sequence[str],
    scenario_hash: str,
    summary_hash: str,
) -> ProbeResult:
    """Assemble the context for one probe, ask the model, grade the answer."""
    ctx = await assemble(
        turns, fact.probe, arm.config, deps,
        system_prompt=system_prompt, pins=pins,
        instruction=config.ANSWER_INSTRUCTION,
    )
    _assert_grep_agrees(ctx, fact)

    present: bool | None = None
    if fact.zone != "negative":
        present = matches(fact.value, ctx.text())

    retrieval_hit: bool | None = None
    if arm.config.enabled("retrieve") and ctx.retrieved is not None:
        retrieval_hit = fact.turn in ctx.retrieved.turn_ids

    payload = ctx.messages()
    result = await llm.complete(
        payload,
        model=model,
        cache_key=cache_key(arm, fact, model, scenario_hash, summary_hash),
        local_prompt_tokens=ctx.used(),
    )
    status, answer = classify(result.text, result.finish_reason, fact, error=result.error)

    return ProbeResult(
        arm=arm.name,
        model=model,
        probe_id=fact.id,
        zone=fact.zone,
        carrier=fact.carrier,
        status=status,
        answer=answer,
        gold=fact.value,
        fact_present=present,
        assembled_tokens=ctx.used(),
        turns_in_context=ctx.turns_in_context(),
        retrieval_hit=retrieval_hit,
        usage=result.usage or Usage(),
        replayed=result.replayed,
        error=result.error,
        layer_trace=tuple(t.as_dict() for t in ctx.traces),
    )


if __name__ == "__main__":
    import asyncio

    from rich.console import Console

    from src import summarizer
    from src.ladder import BY_NAME
    from src.pipeline import make_deps
    from src.scenario import assign_zones, build, spec_hash, window_reach

    console = Console()
    turns, facts, spec = build()
    facts = assign_zones(facts, window_reach(turns))  # sync: not in a loop yet
    fixture = summarizer.load() or {}
    deps = make_deps(
        summarize_fn=summarizer.as_fn(fixture) if fixture else None,
        fudge=config.TOKENIZER_FUDGE,
    )

    async def _smoke() -> None:
        deep = [f for f in facts if f.zone == "deep"][0]
        recent = [f for f in facts if f.zone == "recent"][0]
        for arm_name in ("raw", "full"):
            for fact in (recent, deep):
                r = await run_probe(
                    turns, fact, BY_NAME[arm_name], deps,
                    model=config.CHAT_MODEL,
                    system_prompt=spec["system_prompt"], pins=spec["pinned_facts"],
                    scenario_hash=spec_hash(), summary_hash=str(fixture.get("hash", "")),
                )
                console.print(
                    f"{r.arm:<7} {r.probe_id} {r.zone:<7} "
                    f"present={str(r.fact_present):<5} {r.status:<12} "
                    f"tok={r.assembled_tokens:<5} turns={r.turns_in_context:<3} "
                    f"{('answer=' + r.answer[:30]) if r.answer else r.error[:60]}"
                )
        console.print(
            "\nthe raw rows must read `oversized` - a structural break, not a "
            "0% recall result"
        )

    asyncio.run(_smoke())
