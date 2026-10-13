"""Settings for the failover router: API keys, model IDs and resilience knobs.

Inputs:  .env in the project root (GEMINI_API_KEY, GROQ_API_KEY).
Outputs: module-level constants imported by every other file.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Provider order is the failover order: primary first, last resort last.
GEMINI_MODEL = "gemini-3.5-flash-lite"
GROQ_MODEL = "openai/gpt-oss-20b"
OLLAMA_MODEL = "llama3.2:3b"
PROVIDER_ORDER = ["gemini", "groq", "ollama"]

# Retry: how hard we push one provider before moving to the next.
CALL_TIMEOUT_S = 30.0  # one attempt, one provider
MAX_RETRIES = 2  # extra attempts after the first one
BACKOFF_BASE_S = 0.5  # 0.5s, 1s, 2s ... before jitter
BACKOFF_CAP_S = 8.0  # longer waits than this mean "fail over instead"

# Circuit breaker: when we stop calling a provider at all for a while.
BREAKER_THRESHOLD = 2  # consecutive failed calls that open the breaker
BREAKER_COOLDOWN_S = 30.0  # how long it stays open before one probe call
