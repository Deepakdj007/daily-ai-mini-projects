"""The one interface the agent sees: `await router.complete(messages)`.

Inputs:  chat history (common shape, see providers.py) and the providers in order.
Outputs: a Reply - text plus which provider answered, attempts, failovers, latency.
         It never raises: if every provider is down it returns a degraded Reply.
"""

import time
from dataclasses import dataclass

from src import config
from src.errors import ProviderError
from src.providers import Messages, Provider
from src.resilience import CircuitBreaker, retry_with_backoff

# What the customer gets when nothing can answer: honest, safe, and a human follows up.
DEGRADED_TEXT = ('{"category": "other", "priority": "high", "reply": "Thanks for writing in. '
                 'A member of our team will get back to you within a few hours."}')


@dataclass
class Reply:
    """One answer, plus the operator-only story of how it was produced."""

    text: str
    provider: str  # "gemini" | "groq" | "ollama" | "degraded"
    attempts: int
    failovers: int
    latency_s: float


class LLMRouter:
    """Walk providers in order; retry the retryable, skip the broken, never crash."""

    def __init__(self, providers: list[Provider], cooldown_s: float = config.BREAKER_COOLDOWN_S) -> None:
        self.providers = providers
        self.breakers = {p.name: CircuitBreaker(config.BREAKER_THRESHOLD, cooldown_s) for p in providers}

    def _log(self, line: str) -> None:
        """Operator log. The customer never sees these lines."""
        print(f"    [router] {line}")

    async def _try(self, provider: Provider, messages: Messages, counter: list[int]) -> str:
        """One provider, with retries. Counts every attempt into counter[0]."""
        async def attempt() -> str:
            counter[0] += 1
            return await provider.complete(messages)

        def on_retry(err: ProviderError, delay: float) -> None:
            self._log(f"{provider.name}: {err.reason} retryable -> retry in {delay:.1f}s")

        return await retry_with_backoff(attempt, max_retries=config.MAX_RETRIES, base=config.BACKOFF_BASE_S,
                                        cap=config.BACKOFF_CAP_S, on_retry=on_retry)

    async def complete(self, messages: Messages) -> Reply:
        """Get a reply from the first provider that can give one."""
        start = time.perf_counter()
        counter = [0]
        failovers = 0
        for provider in self.providers:
            breaker = self.breakers[provider.name]
            if not breaker.allow():
                self._log(f"{provider.name}: breaker OPEN -> skipped ({breaker.seconds_left():.0f}s left)")
                failovers += 1
                continue
            if breaker.state == "half_open":
                self._log(f"{provider.name}: breaker HALF-OPEN -> sending one probe call")
            try:
                text = await self._try(provider, messages, counter)
            except ProviderError as err:
                breaker.record_failure()
                kind = "retryable, out of retries" if err.retryable else "fatal, no retry"
                self._log(f"{provider.name}: {err.reason} {kind} -> failover"
                          f"{' (breaker now OPEN)' if breaker.state == 'open' else ''}")
                failovers += 1
                continue
            breaker.record_success()
            return Reply(text, provider.name, counter[0], failovers, time.perf_counter() - start)
        self._log("every provider failed -> degraded reply, flagged for a human")
        return Reply(DEGRADED_TEXT, "degraded", counter[0], failovers, time.perf_counter() - start)
