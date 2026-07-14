"""Central configuration: model choice, loop limits, paths, and the system prompt.

No API keys live here — the whole project runs against a local Ollama server.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Mandatory on Windows: `uv run` does not inherit shell env vars from .env.
load_dotenv()

# qwen3:4b is the smallest model that chains tool calls dependably (~2.6 GB, runs on
# CPU). llama3.2:3b is lighter (~2 GB) but fumbles multi-step tool schemas; qwen3:1.7b
# works if RAM is very tight, with the same caveat.
MODEL = "qwen3:4b"

# Caps the think->tool->observe loop so a confused model can't spin forever.
MAX_STEPS = 6

ROOT_DIR = Path(__file__).resolve().parent.parent

# The only folder the agent is allowed to touch. Created on first tool import.
SANDBOX_DIR = ROOT_DIR / "sandbox"

# Optional override for a non-default Ollama server (None -> localhost:11434).
OLLAMA_HOST = os.environ.get("OLLAMA_HOST") or None

SYSTEM_PROMPT = """You are a helpful offline assistant running locally on the user's laptop.

You have tools for arithmetic, saving/reading/listing notes, the current date and time,
and counting words. Always use a tool for these jobs instead of guessing — your own
math and clock are unreliable.

Rules:
- Never say you saved, read, or calculated something unless a tool result confirms it.
- If the task has steps left after a tool result, call the next tool immediately.
- When every step is done, give one short final answer using the tool results."""
