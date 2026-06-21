"""Shared result wrapper + error explainer for every data client.

Both the football-data.org client and the news client return `ApiResult` so a
failed call degrades into a clean error string instead of raising — one source
going down never crashes the graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import httpx

T = TypeVar("T")


@dataclass
class ApiResult(Generic[T]):
    """Either `data` (success) or `error` (a clean, human-readable reason).

    `transient` flags failures worth retrying (timeout / rate limit) versus
    permanent ones (bad token, resource not found, not configured).
    """

    data: T | None = None
    error: str | None = None
    transient: bool = False

    @property
    def ok(self) -> bool:
        """True when the call succeeded and data is present."""
        return self.error is None and self.data is not None


def is_transient(exc: Exception) -> bool:
    """True when the failure is worth a retry: a timeout, 429, or 5xx."""
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return False


def explain_error(exc: Exception) -> str:
    """Turn a raw exception into a short, user-facing reason string."""
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        reasons = {
            403: "token rejected or resource not on your plan (403)",
            404: "resource not found — data may not be published yet (404)",
            429: "rate limit hit (429)",
        }
        return reasons.get(code, f"HTTP {code}")
    if isinstance(exc, httpx.TimeoutException):
        return "request timed out"
    return f"request failed: {exc}"
