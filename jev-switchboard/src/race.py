"""The race: the same messages classified by Jev and by an LLM acting as the router.

Inputs:  a list of customer messages.
Outputs: per-message intent and latency for both routers, printed side by side.
"""

import json
import statistics
import time

from typesafe_sdk import AsyncTypeSafeClient

from src import config, jev_router
from src.handlers import groq

INTENTS = list(jev_router.QUESTIONS["intent"].criteria)

LLM_ROUTER_PROMPT = f"""Classify the customer message for an online store.
Pick exactly one intent from: {", ".join(INTENTS)}.
Also give your confidence from 0 to 1."""

SCHEMA = {
    "name": "route",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": INTENTS},
            "confidence": {"type": "number"},
        },
        "required": ["intent", "confidence"],
        "additionalProperties": False,
    },
}


async def llm_classify(message: str) -> tuple[str, float, float]:
    """Classify with an LLM. Returns (intent, self-reported confidence, ms)."""
    start = time.perf_counter()
    response = await groq.chat.completions.create(
        model=config.SMALL_LLM,
        messages=[{"role": "system", "content": LLM_ROUTER_PROMPT},
                  {"role": "user", "content": message}],
        response_format={"type": "json_schema", "json_schema": SCHEMA},
        temperature=0,
    )
    ms = (time.perf_counter() - start) * 1000
    answer = json.loads(response.choices[0].message.content)
    return answer["intent"], answer["confidence"], ms


async def run_race(client: AsyncTypeSafeClient, messages: list[str]) -> None:
    """Run both routers on each message, one after the other, and print the results."""
    jev_times, llm_times, agree = [], [], 0
    print(f"{'message':<46} {'jev':<24} {'llm':<24}")
    for message in messages:
        s = await jev_router.classify(client, message)
        llm_intent, llm_conf, llm_ms = await llm_classify(message)
        jev_times.append(s.ms)
        llm_times.append(llm_ms)
        agree += s.intent == llm_intent
        jev_cell = f"{s.intent} {s.intent_confidence:.2f} {s.ms:.0f}ms"
        llm_cell = f"{llm_intent} {llm_conf:.2f} {llm_ms:.0f}ms"
        print(f"{message[:45]:<46} {jev_cell:<24} {llm_cell:<24}")
    print(f"\nmedian latency  jev {statistics.median(jev_times):.0f} ms"
          f"  |  llm {statistics.median(llm_times):.0f} ms")
    print(f"same intent on {agree}/{len(messages)} messages")
