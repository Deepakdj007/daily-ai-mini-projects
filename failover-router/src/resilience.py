"""Retry with backoff, and a circuit breaker. No provider code in here.

Inputs:  an async function that may raise ProviderError.
Outputs: its result, or the last ProviderError once retrying stops making sense.
"""

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from src.errors import ProviderError

T = TypeVar("T")


def backoff_delay(attempt: int, base: float, cap: float,
                  retry_after: float | None = None) -> float | None:
    """Seconds to wait before retry number `attempt` (0-based), or None = give up.

    If the server said how long to wait, believe it - unless that is longer than
    `cap`, in which case waiting is worse than failing over. Otherwise use
    exponential backoff with full jitter: a random wait between 0 and base * 2^n,
    so many clients that failed together do not all retry together.
    """
    if retry_after is not None:
        return retry_after if retry_after <= cap else None
    return random.uniform(0, min(cap, base * 2**attempt))


async def retry_with_backoff(fn: Callable[[], Awaitable[T]], *, max_retries: int,
                             base: float, cap: float,
                             on_retry: Callable[[ProviderError, float], None]) -> T:
    """Call `fn`; retry only retryable errors, at most `max_retries` extra times."""
    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except ProviderError as err:
            if not err.retryable or attempt == max_retries:
                raise
            delay = backoff_delay(attempt, base, cap, err.retry_after)
            if delay is None:
                raise
            on_retry(err, delay)
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")


class CircuitBreaker:
    """Stops calling a provider that keeps failing, then tests it again later.

    closed    -> calls go through. `threshold` failures in a row -> open.
    open      -> calls are skipped instantly, no time wasted. After `cooldown_s`
                 the next caller gets through as a probe -> half_open.
    half_open -> the probe decides: success -> closed, failure -> open again.
    """

    def __init__(self, threshold: int, cooldown_s: float,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self.clock = clock  # injectable, so the self-test can fast-forward time
        self.state = "closed"
        self.failures = 0
        self.opened_at = 0.0

    def allow(self) -> bool:
        """True if a call may go through right now."""
        if self.state == "open" and self.seconds_left() <= 0:
            self.state = "half_open"
        return self.state != "open"

    def seconds_left(self) -> float:
        """How long until an open breaker lets a probe through."""
        return max(0.0, self.opened_at + self.cooldown_s - self.clock())

    def record_success(self) -> None:
        """A call worked: forget past failures."""
        self.state = "closed"
        self.failures = 0

    def record_failure(self) -> None:
        """A call failed (after its retries): maybe open the breaker."""
        self.failures += 1
        if self.state == "half_open" or self.failures >= self.threshold:
            self.state = "open"
            self.opened_at = self.clock()
