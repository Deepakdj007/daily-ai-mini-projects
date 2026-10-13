"""Settings for the RAG debugger: keys, paths, models and the retrieval policy.

Inputs:  .env (GROQ_API_KEY)
Outputs: constants imported by every other module
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# uv run does not pass .env through on Windows, so load it before anything reads a key.
load_dotenv()

# Groq replies contain characters like U+2011 that crash the default cp1252 console.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT_DIR / "data" / "kumkum_home.json"
QUERIES_PATH = ROOT_DIR / "data" / "queries.json"
TRACE_DIR = ROOT_DIR / "output" / "traces"

# The retrieval policy under test: what a plain RAG pipeline would do.
TOP_K = 3              # passages that reach the prompt
MIN_SIMILARITY = 0.30  # anything weaker is cut even inside the top-k

# The debugger looks further than the pipeline does. A passage ranked 7th
# never reaches the prompt, but the judge still reads it - that is how a
# miss becomes visible.
JUDGE_POOL = 10

JUDGE_MODEL = "openai/gpt-oss-120b"
ANSWER_MODEL = "openai/gpt-oss-120b"
JUDGE_MAX_TOKENS = 2000   # gpt-oss spends completion tokens reasoning before the JSON
ANSWER_MAX_TOKENS = 600


def require_api_key() -> str:
    """Return the Groq key, or stop with a message that says how to get one."""
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "GROQ_API_KEY is missing. Create a free key at https://console.groq.com/keys "
            "and put it in .env as GROQ_API_KEY=..."
        )
    return key


if __name__ == "__main__":
    print(f"corpus   {CORPUS_PATH} exists={CORPUS_PATH.exists()}")
    print(f"policy   top-k={TOP_K} min-sim={MIN_SIMILARITY} judge-pool={JUDGE_POOL}")
    print(f"models   judge={JUDGE_MODEL} answer={ANSWER_MODEL}")
    print(f"groq key {'set' if os.getenv('GROQ_API_KEY') else 'MISSING'}")
