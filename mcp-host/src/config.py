"""
Central configuration for the MCP host.

Loads the Groq API key from .env, pins the model, and resolves every path the
host and its child servers need.

Inputs: environment variables from .env.
Outputs: module-level constants, require_keys(), make_client().
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from groq import AsyncGroq

# uv run does not inherit .env on Windows, so load it before anything reads os.environ.
load_dotenv()

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
SERVERS_JSON: Path = PROJECT_ROOT / "servers.json"
WORKSPACE_DIR: Path = PROJECT_ROOT / "workspace"

GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")

# openai/gpt-oss-120b is a free-tier Groq production model with a 131k context
# window and native parallel tool calling. llama-3.3-70b-versatile is not an
# option here — it shuts down on 2026-08-16.
MODEL: str = "openai/gpt-oss-120b"

MAX_STEPS: int = 10             # tool-calling rounds before we force a final answer
TOOL_TIMEOUT: float = 120.0     # seconds to wait on a single MCP tool call
CONNECT_TIMEOUT: float = 180.0  # seconds to boot all servers on a cold start
MAX_RESULT_CHARS: int = 6000    # truncate tool output so a web page can't eat the context

SYSTEM_PROMPT: str = (
    "You are an MCP host with tools drawn from five independent servers. "
    "Tool names are namespaced as server__tool, so files__write_file lives on the "
    "'files' server. Chain tools across servers freely to answer a request, and "
    "call independent tools in the same turn rather than one at a time. "
    "File paths are relative to the files server's sandbox root — pass "
    "'summary.md', never 'workspace/summary.md'. "
    "When a tool result starts with ERROR:, read it and try a corrected call. "
    "Answer in plain prose, and say which servers you used."
)


def require_keys() -> None:
    """Fail fast with an actionable message if the Groq key is missing."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and paste a free key "
            "from https://console.groq.com/keys"
        )


def make_client() -> AsyncGroq:
    """Build the async Groq client. Free-tier traffic needs retries for 429s."""
    require_keys()
    return AsyncGroq(api_key=GROQ_API_KEY, max_retries=4)
