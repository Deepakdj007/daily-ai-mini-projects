"""The two model tiers, behind one identical interface.

Inputs:  a query string.
Outputs: {answer, in_tok, out_tok, cost_usd, latency_s, server_s, model, tier}

Both callers return the same dict so every downstream strategy can treat a
local 4B and a hosted 120B as interchangeable. That uniformity is the whole
reason a router is even expressible.
"""

from __future__ import annotations

import time
from typing import Any

from src import cache
from src.config import (
    GROQ_API_KEY,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    OLLAMA_HOST,
    SELF_CONSISTENCY_TEMP,
    SLM_GROQ_MODEL,
    SLM_MAX_TOKENS,
    SLM_OLLAMA_MODEL,
    SLM_PROVIDER,
    SYSTEM_PROMPT,
    TEMPERATURE,
)
from src.pricing import calculate_cost_usd

_groq_client = None
_ollama_client = None


def _groq() -> Any:
    """Lazily build the Groq client so importing this module needs no key."""
    global _groq_client
    if _groq_client is None:
        from groq import Groq

        _groq_client = Groq(api_key=GROQ_API_KEY, max_retries=4)
    return _groq_client


def _ollama() -> Any:
    global _ollama_client
    if _ollama_client is None:
        from ollama import Client

        _ollama_client = Client(host=OLLAMA_HOST)
    return _ollama_client


def _messages(query: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]


def call_groq(model: str, query: str, max_tokens: int, temperature: float) -> dict[str, Any]:
    """One Groq chat completion, with token counts and server-side timing."""
    started = time.perf_counter()
    response = _groq().chat.completions.create(
        model=model,
        messages=_messages(query),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    latency = time.perf_counter() - started

    usage = response.usage
    in_tok = usage.prompt_tokens or 0
    out_tok = usage.completion_tokens or 0
    # total_time is Groq's own inference time. Wall-clock from India includes a
    # transatlantic round trip and free-tier queueing, so reporting only
    # wall-clock would blame the model for the network.
    server_s = float(getattr(usage, "total_time", 0.0) or 0.0)

    return {
        "answer": response.choices[0].message.content or "",
        "in_tok": in_tok,
        "out_tok": out_tok,
        "cost_usd": calculate_cost_usd(model, in_tok, out_tok),
        "latency_s": latency,
        "server_s": server_s,
        "model": model,
    }


def call_ollama(model: str, query: str, max_tokens: int, temperature: float) -> dict[str, Any]:
    """One local Ollama chat call.

    think=False matters twice over. It removes the latency of a reasoning
    preamble, and it stops a <think> block leaking into text that gets graded.
    The grader strips leaked blocks anyway, because trusting a flag you have
    not verified is how measurement projects quietly break.
    """
    started = time.perf_counter()
    response = _ollama().chat(
        model=model,
        messages=_messages(query),
        think=False,
        options={"temperature": temperature, "num_predict": max_tokens},
    )
    latency = time.perf_counter() - started

    # These fields are Optional in the ollama client and are None often enough
    # to matter - guard before arithmetic or the cost ledger throws mid-run.
    in_tok = response.prompt_eval_count or 0
    out_tok = response.eval_count or 0
    total_duration = response.total_duration or 0

    return {
        "answer": response.message.content or "",
        "in_tok": in_tok,
        "out_tok": out_tok,
        "cost_usd": 0.0,
        "latency_s": latency,
        "server_s": total_duration / 1e9,
        "model": f"local:{model}",
    }


def _cached(tier: str, model: str, query: str, sample: int, produce) -> dict[str, Any]:
    """Return a cached call if present, otherwise produce and store one."""
    key = cache.make_key(tier, model, query, sample)
    hit = cache.get(key)
    if hit is not None:
        hit["cached"] = True
        return hit
    result = produce()
    result["tier"] = tier
    result["cached"] = False
    cache.put(key, result)
    return result


def call_llm(query: str) -> dict[str, Any]:
    """The expensive tier. Always Groq gpt-oss-120b."""
    return _cached(
        "llm",
        LLM_MODEL,
        query,
        0,
        lambda: call_groq(LLM_MODEL, query, LLM_MAX_TOKENS, TEMPERATURE),
    )


def call_slm(query: str, sample: int = 0) -> dict[str, Any]:
    """The cheap tier. Local Ollama by default, Groq gpt-oss-20b in tier D.

    sample > 0 requests an independent draw at a higher temperature, which is
    what the verifier's self-consistency check consumes.
    """
    temperature = TEMPERATURE if sample == 0 else SELF_CONSISTENCY_TEMP

    if SLM_PROVIDER == "groq":
        model = SLM_GROQ_MODEL
        produce = lambda: call_groq(model, query, SLM_MAX_TOKENS, temperature)
    else:
        model = SLM_OLLAMA_MODEL
        produce = lambda: call_ollama(model, query, SLM_MAX_TOKENS, temperature)

    return _cached("slm", model, query, sample, produce)


def preflight() -> None:
    """Fail early with actionable messages instead of deep inside a long run."""
    if SLM_PROVIDER == "ollama":
        try:
            _ollama().list()
        except Exception as exc:  # noqa: BLE001 - message quality matters here
            raise SystemExit(
                f"Cannot reach Ollama at {OLLAMA_HOST or 'http://localhost:11434'}: {exc}\n"
                "Start it with: ollama serve\n"
                f"Then pull the model: ollama pull {SLM_OLLAMA_MODEL}\n"
                "No Ollama at all? Run with SLM_PROVIDER=groq to use gpt-oss-20b instead."
            ) from exc


if __name__ == "__main__":
    from src.config import require_keys, slm_label

    require_keys()
    preflight()
    question = "A shop sells pens at 12 rupees each. What do 7 pens cost?"

    print(f"slm tier: {slm_label()}")
    slm = call_slm(question)
    print(f"  answer   : {repr(slm['answer'])[:90]}")
    print(f"  tokens   : in={slm['in_tok']} out={slm['out_tok']}")
    print(f"  cost     : ${slm['cost_usd']:.8f}")
    print(f"  latency  : {slm['latency_s']:.2f}s (server {slm['server_s']:.2f}s)")

    print("\nllm tier: " + str(LLM_MODEL))
    llm = call_llm(question)
    print(f"  answer   : {repr(llm['answer'])[:90]}")
    print(f"  tokens   : in={llm['in_tok']} out={llm['out_tok']}")
    print(f"  cost     : ${llm['cost_usd']:.8f}")
    print(f"  latency  : {llm['latency_s']:.2f}s (server {llm['server_s']:.2f}s)")
