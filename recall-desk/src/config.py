"""Central configuration for recall-desk.

Holds env vars, paths, retrieval budgets, and the MemoryConfig dataclass that
switches individual memory tiers off. Nothing here does I/O beyond reading .env.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _force_utf8_console() -> None:
    """Make stdout survive model output on a Windows terminal.

    Models sprinkle characters like U+202F (narrow no-break space) into replies.
    The default Windows console encoding is cp1252, which cannot encode them, so
    printing a perfectly good answer raises UnicodeEncodeError.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


_force_utf8_console()

# --- LLM -------------------------------------------------------------------
GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
STORE_PATH = ROOT / "recall.db"
CHECKPOINT_PATH = ROOT / "threads.db"

# --- Embeddings ------------------------------------------------------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DIMS = 384  # all-MiniLM-L6-v2 output width; must match the store's index config

# --- Retrieval budgets -----------------------------------------------------
K_SEM = 4  # semantic facts injected per turn
K_EPI = 2  # past episodes injected per turn
MAX_RULES = 6  # hard cap on the playbook, so the always-on tier stays cheap

# Relevance floor for episodic recall only. The two searched tiers are not
# symmetric: the episode corpus grows without bound and most of it is irrelevant
# to any given ticket, so a weak match is noise worth dropping. A customer's
# account record is small, bounded, and almost always worth including, so
# semantic recall has no floor — a floor there could drop the plan fact on a
# ticket that never says the word "plan".
EPISODE_MIN_SCORE = 0.25

# --- Customers -------------------------------------------------------------
CUSTOMERS: dict[str, str] = {
    "acme": "Acme Retail",
    "beta": "Beta Corp",
}


@dataclass(frozen=True)
class MemoryConfig:
    """Which memory tiers the recall step is allowed to read.

    This is the entire ablation mechanism. The harness flips these three
    booleans and changes nothing else, so a scorecard difference can only come
    from the tier that was switched off.
    """

    semantic: bool = True
    episodic: bool = True
    procedural: bool = True

    @property
    def label(self) -> str:
        """Short name used in the ablation scorecard."""
        off = [
            name
            for name, on in (
                ("semantic", self.semantic),
                ("episodic", self.episodic),
                ("procedural", self.procedural),
            )
            if not on
        ]
        if not off:
            return "all on"
        if len(off) == 3:
            return "no memory"
        return "no " + "+".join(off)


ALL_ON = MemoryConfig()
NO_SEMANTIC = MemoryConfig(semantic=False)
NO_EPISODIC = MemoryConfig(episodic=False)
NO_PROCEDURAL = MemoryConfig(procedural=False)
NO_MEMORY = MemoryConfig(False, False, False)

ABLATION_CONFIGS = [ALL_ON, NO_SEMANTIC, NO_EPISODIC, NO_PROCEDURAL, NO_MEMORY]


def require_api_key() -> str:
    """Return the Groq key, or raise with a pointer to where to get one."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and paste a free "
            "key from https://console.groq.com/keys"
        )
    return GROQ_API_KEY
