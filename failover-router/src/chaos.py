"""Fault injection: make providers fail on cue, so the demo does not wait for a real outage.

Inputs:  CLI specs like "gemini@3-4" (down for tickets 3 and 4), "gemini,groq"
         (down for the whole run) and a set of flaky providers (fail once).
Outputs: a hook on each provider that raises that SDK's own exception type, so
         errors.py, the retries and the breakers all run exactly as in a real outage.
"""

import groq
import httpx
import ollama
from google.genai import errors as genai_errors

from src.providers import Provider


def _sdk_error(provider: str, status: int) -> Exception:
    """Build the exception the real SDK would raise for this status code."""
    if provider == "gemini":
        label = {429: "RESOURCE_EXHAUSTED", 400: "INVALID_ARGUMENT"}.get(status, "UNAVAILABLE")
        body = {"error": {"code": status, "message": "chaos: injected failure", "status": label}}
        cls = genai_errors.ClientError if status < 500 else genai_errors.ServerError
        return cls(status, body)
    if provider == "groq":
        response = httpx.Response(status, request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"))
        cls = {429: groq.RateLimitError, 401: groq.AuthenticationError}.get(status, groq.InternalServerError)
        return cls("chaos: injected failure", response=response, body=None)
    if status == 503:
        return ollama.ResponseError("chaos: injected failure", status)
    return ConnectionError("Failed to connect to Ollama (chaos: injected)")


def _parse_steps(spec: str) -> set[int] | None:
    """'3-4' -> {3, 4}; '5' -> {5}; '' -> None, meaning every ticket."""
    if not spec:
        return None
    low, _, high = spec.partition("-")
    return set(range(int(low), int(high or low) + 1))


class ChaosPlan:
    """Which providers fail, and on which tickets."""

    def __init__(self, kill: str = "", flaky: str = "") -> None:
        self.down: dict[str, set[int] | None] = {}
        for part in filter(None, kill.split(",")):
            name, _, steps = part.partition("@")
            self.down[name] = _parse_steps(steps)
        self.flaky = set(filter(None, flaky.split(",")))
        self.step = 0

    def begin_ticket(self, step: int) -> None:
        """Called by the agent loop before each ticket."""
        self.step = step

    def inject(self, provider: str) -> None:
        """Raise a fake outage for `provider` if the plan says so; else do nothing."""
        if provider in self.flaky:
            self.flaky.discard(provider)  # fail exactly once, then behave
            raise _sdk_error(provider, 503)
        if provider in self.down:
            steps = self.down[provider]
            if steps is None or self.step in steps:
                raise _sdk_error(provider, 429 if provider != "ollama" else 0)

    def attach(self, providers: list[Provider]) -> None:
        """Install the hook on every provider."""
        for provider in providers:
            provider.chaos = self.inject
