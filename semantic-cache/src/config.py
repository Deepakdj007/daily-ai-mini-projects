"""Settings for the semantic cache demo.

Inputs: GEMINI_API_KEY from .env.
Outputs: constants used by every other module (model, prices, threshold, paths).
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Windows consoles default to cp1252, which can't print every character a model writes.
sys.stdout.reconfigure(encoding="utf-8")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL = "gemini-3.5-flash-lite"

# The free tier allows 15 requests per minute for this model; space calls out.
MIN_INTERVAL_S = 4.2

# Paid-tier price per 1M tokens (USD). The free tier costs nothing; these
# numbers show what the same traffic would cost once you pay for it.
PRICE_IN_PER_M = 0.30
PRICE_OUT_PER_M = 2.50

# A cached answer is reused only when the new question is at least this similar.
THRESHOLD = 0.82

# Entries older than this many seconds are treated as missing.
TTL_SECONDS = 24 * 60 * 60

CACHE_DIR = ROOT / ".cache"
COLLECTION = "answers"
