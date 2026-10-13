"""The same agent run, with LiteLLM's Router doing the failover instead of our code.

Inputs:  the same CLI flags as `run` (--kill, --bad-key, --cooldown).
Outputs: the same tickets and summary, so the two routers can be compared side by side.
"""

import argparse
import time

import litellm
from litellm import Router

from src import config
from src.chaos import _parse_steps
from src.main import print_summary
from src.router import DEGRADED_TEXT
from src.tickets import SYSTEM_PROMPT, TICKETS, parse_triage

litellm.suppress_debug_info = True


def build_router(args: argparse.Namespace) -> Router:
    """Three model groups, a fallback chain and the retry/cooldown knobs - all config."""
    keys = {"gemini": config.GEMINI_API_KEY, "groq": config.GROQ_API_KEY}
    if args.bad_key:
        keys[args.bad_key] = "not-a-real-key"
    model_list = [
        {"model_name": "gemini", "litellm_params": {"model": f"gemini/{config.GEMINI_MODEL}", "api_key": keys["gemini"]}},
        {"model_name": "groq", "litellm_params": {"model": f"groq/{config.GROQ_MODEL}", "api_key": keys["groq"]}},
        {"model_name": "ollama", "litellm_params": {"model": f"ollama_chat/{config.OLLAMA_MODEL}"}},
    ]
    return Router(
        model_list=model_list,
        fallbacks=[{"gemini": ["groq", "ollama"]}],  # the failover order
        num_retries=config.MAX_RETRIES,  # retries per model group, with backoff
        timeout=config.CALL_TIMEOUT_S,
        allowed_fails=config.BREAKER_THRESHOLD - 1,  # failures tolerated before cooldown
        cooldown_time=args.cooldown,  # LiteLLM's version of an open breaker
    )


def set_down(router: Router, provider: str, down: bool) -> None:
    """Chaos for LiteLLM: make one deployment raise a real litellm 429 (or stop)."""
    for deployment in router.model_list:
        if deployment["model_name"] == provider:
            params = deployment["litellm_params"]
            if down:
                params["mock_response"] = "litellm.RateLimitError"
            else:
                params.pop("mock_response", None)


def provider_of(model: str) -> str:
    """Turn the model ID LiteLLM reports back into our provider name."""
    names = {config.GEMINI_MODEL: "gemini", config.GROQ_MODEL: "groq", config.OLLAMA_MODEL: "ollama"}
    return next((name for model_id, name in names.items() if model_id in model), model)


async def ask(router: Router, history: list[dict[str, str]]) -> tuple[str, str, dict]:
    """One LiteLLM call; if the whole chain fails, degrade exactly like our router."""
    try:
        response = await router.acompletion(model="gemini", messages=history,
                                            response_format={"type": "json_object"})
    except Exception as exc:
        print(f"    [litellm] every provider failed ({type(exc).__name__}) -> degraded reply")
        return DEGRADED_TEXT, "degraded", {}
    headers = response._hidden_params.get("additional_headers") or {}
    return response.choices[0].message.content or "", provider_of(response.model), headers


async def run_litellm(args: argparse.Namespace) -> None:
    """Triage the same tickets through LiteLLM's Router."""
    router = build_router(args)
    killed = {}
    for part in filter(None, args.kill.split(",")):
        name, _, steps = part.partition("@")
        killed[name] = _parse_steps(steps)
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    results = []
    for step, ticket in enumerate(TICKETS, start=1):
        for name, steps in killed.items():
            set_down(router, name, steps is None or step in steps)
        print(f"\nTicket {step}: {ticket}")
        history.append({"role": "user", "content": ticket})
        start = time.perf_counter()
        text, provider, headers = await ask(router, history)
        if provider == "degraded":
            history.pop()
        else:
            history.append({"role": "assistant", "content": text})
        print(f"  Customer sees: {parse_triage(text)['reply']}")
        results.append({"step": step, "provider": provider,
                        "attempts": 1 + headers.get("x-litellm-attempted-retries", 0),
                        "failovers": headers.get("x-litellm-attempted-fallbacks", 0),
                        "latency_s": time.perf_counter() - start, "needs_human": provider == "degraded"})
    print_summary(results)
