"""Settings for the CRAG help desk: keys, models and retrieval knobs.

Inputs:  .env (GROQ_API_KEY)
Outputs: constants imported by every other module
"""

import os
import sys

from dotenv import load_dotenv

# uv run does not pass .env through on Windows, so load it before anything reads a key.
load_dotenv()

# Groq replies contain characters like U+2011 that crash the default cp1252 console.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

TOP_K = 3             # passages the retriever hands to the evaluator
WEB_RESULTS = 4       # web snippets fetched when the KB is not enough

MODEL = "openai/gpt-oss-120b"
GRADE_MAX_TOKENS = 2000   # gpt-oss spends completion tokens reasoning before the JSON
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
    print(f"model    {MODEL}")
    print(f"top-k    {TOP_K}  web results {WEB_RESULTS}")
    print(f"groq key {'set' if os.getenv('GROQ_API_KEY') else 'MISSING'}")
