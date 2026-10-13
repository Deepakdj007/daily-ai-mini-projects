"""One error type for every provider, so the router never imports an SDK.

Inputs:  any exception raised by google-genai, groq, ollama or the network.
Outputs: ProviderError with `retryable` and an optional `retry_after` (seconds).
"""

import groq
import httpx
import ollama
from google.genai import errors as genai_errors

# Status codes worth trying again: rate limits, timeouts and server-side trouble.
# Everything else (400 bad request, 401 bad key, 404 wrong model) fails the same
# way every time, so retrying only burns the clock.
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class ProviderError(Exception):
    """A provider call failed. `retryable` decides retry vs. instant failover."""

    def __init__(self, provider: str, reason: str, status: int | None,
                 retryable: bool, retry_after: float | None = None) -> None:
        super().__init__(f"{provider}: {reason}")
        self.provider = provider
        self.reason = reason  # short label for logs: "429" or "ConnectionError(...)"
        self.status = status
        self.retryable = retryable
        self.retry_after = retry_after


def _gemini_retry_after(err: genai_errors.APIError) -> float | None:
    """Read Gemini's RetryInfo.retryDelay (e.g. '41s') from the error details."""
    details = err.details.get("error", {}).get("details", []) if isinstance(err.details, dict) else []
    for item in details:
        if item.get("@type", "").endswith("RetryInfo"):
            return float(str(item.get("retryDelay", "0s")).rstrip("s") or 0)
    return None


def _header_retry_after(headers: httpx.Headers) -> float | None:
    """Read a numeric Retry-After header, the way Groq sends it on a 429."""
    value = headers.get("retry-after")
    try:
        return float(value) if value else None
    except ValueError:
        return None


def _status_and_wait(exc: Exception) -> tuple[int | None, float | None, bool]:
    """Map an SDK exception to (status, retry_after, is_network_error)."""
    if isinstance(exc, genai_errors.APIError):
        return exc.code, _gemini_retry_after(exc), False
    if isinstance(exc, groq.APIStatusError):
        return exc.status_code, _header_retry_after(exc.response.headers), False
    if isinstance(exc, ollama.ResponseError):
        return exc.status_code, None, False
    network = (groq.APIConnectionError, httpx.TransportError, ConnectionError, TimeoutError)
    return None, None, isinstance(exc, network)


def _reason(exc: Exception, status: int | None) -> str:
    """'429' for HTTP errors; 'ValueError(GROQ_API_KEY is not set)' for the rest."""
    if status:
        return str(status)
    first_sentence = str(exc).split(".")[0][:50]
    return f"{type(exc).__name__}({first_sentence})" if first_sentence else type(exc).__name__


def to_provider_error(provider: str, exc: Exception) -> ProviderError:
    """Classify any exception from a provider call as retryable or fatal."""
    if isinstance(exc, ProviderError):
        return exc
    status, retry_after, network = _status_and_wait(exc)
    retryable = network or status in RETRYABLE_STATUS
    return ProviderError(provider, _reason(exc, status), status, retryable, retry_after)
