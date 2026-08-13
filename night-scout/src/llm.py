"""The only place the agent talks to Groq.

Inputs:  a pydantic schema + chat messages
Outputs: a validated instance of that schema

Two things live here that are easy to get wrong anywhere else: the rate throttle
(which must run on real time, not simulated time) and the retry ladder.
"""

import time
from typing import Any, Sequence, TypeVar

from pydantic import BaseModel

from src import config

T = TypeVar("T", bound=BaseModel)

_last_call_at: float = 0.0

# Each attempt changes the generation conditions, because a deterministic failure
# would otherwise repeat forever: (temperature, strict). Attempt 1 is the fast
# deterministic path; attempt 2 adds warmth to break a stuck decode; attempt 3
# drops constrained decoding and lets pydantic validate best-effort JSON.
_ATTEMPTS: tuple[tuple[float, bool | None], ...] = ((0.0, True), (0.4, True), (0.4, None))


def throttle() -> None:
    """Block until MIN_SECONDS_BETWEEN_LLM_CALLS of real time has passed.

    Deliberately not scaled by TIME_SCALE. --demo sprints through eight simulated
    hours in three real minutes, but Groq's tokens-per-minute window is measured in
    real minutes and does not fast-forward with us.
    """
    global _last_call_at
    wait = config.MIN_SECONDS_BETWEEN_LLM_CALLS - (time.monotonic() - _last_call_at)
    if wait > 0:
        time.sleep(wait)
    _last_call_at = time.monotonic()


def call_structured(schema: type[T], messages: Sequence[tuple[str, str]]) -> T:
    """Ask Groq for one `schema` instance, throttled and retried.

    method="json_schema" rather than the default tool-calling path: gpt-oss is a
    reasoning model, and on the tool-calling path it emits prose that Groq rejects
    with `400 tool_use_failed`. strict=True turns on constrained decoding, which
    only openai/gpt-oss-20b and -120b support — passing it for any other model is
    silently ignored by langchain-groq.
    """
    if config.STUB_LLM:
        from src import stub

        return stub.respond(schema, messages)

    last_error: Exception | None = None
    for temperature, strict in _ATTEMPTS:
        # Throttle per attempt, not once per call: a retry is another request.
        throttle()
        runnable: Any = config.make_llm(temperature).with_structured_output(
            schema, method="json_schema", strict=strict
        )
        try:
            result = runnable.invoke(list(messages))
        except Exception as exc:  # noqa: BLE001 — the ladder is the handler
            last_error = exc
            continue
        if isinstance(result, schema):
            return result
        last_error = TypeError(f"expected {schema.__name__}, got {type(result).__name__}")
    raise RuntimeError(
        f"{schema.__name__} could not be produced after {len(_ATTEMPTS)} attempts: "
        f"{type(last_error).__name__}: {last_error}"
    ) from last_error
