"""Configuration and constants for the self-healing code agent.

Loads the Groq API key from .env (mandatory on Windows, where `uv run` does not
inherit the .env automatically) and exposes the model name plus the two safety
limits that keep a confused agent from running away.
"""

from dotenv import load_dotenv

# Load .env before anything imports a client, so GROQ_API_KEY is in the env.
load_dotenv()

# Groq's free-tier model. Tool-calling capable, no credit card needed.
MODEL = "llama-3.3-70b-versatile"

# Hard cap on model round-trips for one task. Each write-run-fix cycle costs a
# few requests, so this bounds how many times the agent may try before giving up.
MAX_REQUESTS = 8

# Seconds a single code execution may run before we kill it. Catches infinite
# loops the agent might accidentally write.
EXEC_TIMEOUT = 10
