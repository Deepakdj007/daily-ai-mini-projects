"""Free checks for the resilience logic: no API key, no network, no tokens.

Inputs:  nothing.
Outputs: one PASS/FAIL line per check; returns True if all passed.
"""

from src.chaos import _sdk_error
from src.errors import to_provider_error
from src.resilience import CircuitBreaker, backoff_delay


class FakeClock:
    """A clock you move by hand, so a 30-second cooldown takes no time."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def check_breaker() -> list[tuple[str, bool]]:
    """Walk one breaker through closed -> open -> half_open -> open -> closed."""
    clock = FakeClock()
    b = CircuitBreaker(threshold=2, cooldown_s=30, clock=clock)
    results = [("starts closed", b.allow() and b.state == "closed")]
    b.record_failure()
    results.append(("1 failure: still closed", b.state == "closed"))
    b.record_failure()
    results.append(("2 failures: open, calls blocked", b.state == "open" and not b.allow()))
    clock.now = 29.9
    results.append(("29.9s later: still blocked", not b.allow()))
    clock.now = 30.0
    results.append(("30s later: half-open probe allowed", b.allow() and b.state == "half_open"))
    b.record_failure()
    results.append(("probe fails: open again", b.state == "open" and not b.allow()))
    clock.now = 60.0
    b.allow()
    b.record_success()
    results.append(("probe succeeds: closed", b.state == "closed" and b.failures == 0))
    return results


def check_backoff() -> list[tuple[str, bool]]:
    """Backoff grows, stays under the cap, and respects Retry-After."""
    waits = [backoff_delay(n, base=0.5, cap=8.0) for n in range(10) for _ in range(50)]
    return [
        ("jittered waits stay within [0, cap]", all(0 <= w <= 8.0 for w in waits)),
        ("attempt 0 waits at most 0.5s", all(backoff_delay(0, 0.5, 8.0) <= 0.5 for _ in range(50))),
        ("Retry-After 3s is obeyed", backoff_delay(0, 0.5, 8.0, retry_after=3.0) == 3.0),
        ("Retry-After 40s -> fail over instead", backoff_delay(0, 0.5, 8.0, retry_after=40.0) is None),
    ]


def check_classification() -> list[tuple[str, bool]]:
    """Real SDK exception types map to the right retry decision."""
    def retryable(provider: str, status: int) -> bool:
        return to_provider_error(provider, _sdk_error(provider, status)).retryable

    return [
        ("gemini 429 -> retry", retryable("gemini", 429)),
        ("gemini 503 -> retry", retryable("gemini", 503)),
        ("gemini 400 (bad key) -> no retry", not retryable("gemini", 400)),
        ("groq 429 -> retry", retryable("groq", 429)),
        ("groq 401 (bad key) -> no retry", not retryable("groq", 401)),
        ("ollama not running -> retry", retryable("ollama", 0)),
    ]


def run_selftest() -> bool:
    """Run every check and print the results."""
    checks = check_breaker() + check_backoff() + check_classification()
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    passed = sum(ok for _, ok in checks)
    print(f"\n{passed}/{len(checks)} checks passed")
    return passed == len(checks)
