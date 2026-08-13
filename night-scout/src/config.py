"""Every knob night-scout has, in one file.

Inputs:  .env (GROQ_API_KEY)
Outputs: constants, paths, require_keys(), make_llm()

Edit INTEREST_PROFILE first — it is the only thing that decides what the agent
keeps and what it throws away.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load .env BEFORE reading any variable. On Windows `uv run` does not inherit the
# shell's environment, so this call is mandatory and must come first.
load_dotenv()

GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")

# --- Model. openai/gpt-oss-120b is a free-tier Groq production model with a 131k
# context window and strict structured output. We avoid llama-3.3-70b-versatile:
# it is scheduled to shut down on 2026-08-16.
MODEL: str = "openai/gpt-oss-120b"

# gpt-oss is a reasoning model: it spends tokens thinking before it emits JSON. Too
# small a budget and it runs out mid-thought, returning an empty generation that
# fails validation with `400 json_validate_failed`.
MAX_TOKENS: int = 4096
REASONING_EFFORT: str = "low"  # triage is a ranking job, not a research job

# --- Paths. Derived from this file's location, never from the working directory,
# so the night process and the Streamlit inbox always open the same databases.
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
MEM_PATH: Path = ROOT_DIR / "memory.db"        # LangGraph checkpoints (library-owned)
DB_PATH: Path = ROOT_DIR / "nightscout.db"     # our items, drafts index, wake log
OUTPUT_DIR: Path = ROOT_DIR / "output"
READING_LIST: Path = OUTPUT_DIR / "reading-list.md"

# --- The night. A 23:00-07:00 window at 45-minute cadence is 11 wakes.
WINDOW_START_HOUR: int = 23
WINDOW_END_HOUR: int = 7
WAKE_EVERY_MINUTES: int = 45

# TIME_SCALE compresses simulated time. 1.0 is a real night; the --demo flag swaps
# in DEMO_TIME_SCALE so a 45-minute gap passes in ~11 real seconds and the whole
# night runs in about three minutes.
TIME_SCALE: float = 1.0
DEMO_TIME_SCALE: float = 240.0
MIN_REAL_SLEEP_SECONDS: float = 0.5

# Deliberately NOT scaled by TIME_SCALE. The simulated clock can sprint through a
# night; Groq's tokens-per-minute window cannot.
MIN_SECONDS_BETWEEN_LLM_CALLS: float = 15.0

# --- Budget. Free tier is 30 RPM / 8,000 TPM / 200,000 TPD, and TPD is counted
# across the whole account, not per key. These caps keep a full night near 25k.
MAX_ITEMS_PER_WAKE: int = 12        # items entering one batched triage call
MAX_DEEP_PASSES_PER_WAKE: int = 1   # drafts per wake — spreads load across the night
MAX_DEEP_PASSES_PER_NIGHT: int = 6  # whole-night ceiling
SCORE_THRESHOLD: int = 7            # 0-10; at or above this an item earns a draft
TRIAGE_SNIPPET_CHARS: int = 220     # per item, into the batched call
DETAIL_MAX_CHARS: int = 4000        # article text into the draft call
MAX_REVISIONS: int = 2              # 'edit' re-drafts before we stop offering it

# --- Sources. All keyless.
HN_LOOKBACK_HOURS: int = 6   # a rolling window, not "since the last wake": a story
HN_MIN_POINTS: int = 20      # needs hours to reach 20 points, and dedup makes
HN_HITS_PER_PAGE: int = 20   # re-polling the same items free.

ARXIV_CATEGORIES: tuple[str, ...] = ("cs.AI", "cs.LG")
ARXIV_MAX_RESULTS: int = 15
ARXIV_DELAY_SECONDS: float = 3.0  # arXiv's terms of use: one request every 3 seconds
ARXIV_COOLDOWN_SECONDS: float = 900.0  # how long to leave arXiv alone after a 429

RSS_FEEDS: tuple[str, ...] = (
    "https://simonwillison.net/atom/everything/",
    "https://huggingface.co/blog/feed.xml",
)
# The Hugging Face feed serves its entire archive back to 2021, so each feed is
# truncated to its newest entries before anything else looks at them.
RSS_ENTRIES_PER_FEED: int = 15

# Nothing older than this is news. Applied to every source, and it is what keeps a
# feed's back catalogue out of the night.
MAX_ITEM_AGE_HOURS: int = 48

USER_AGENT: str = "night-scout/0.1 (+https://github.com/datasciencebrain)"
HTTP_TIMEOUT: float = 20.0

# --- SQLite. Two processes share these files, so readers must be willing to wait
# for a writer instead of failing instantly.
SQLITE_TIMEOUT: float = 30.0

# --- What the agent is looking for. This is the prompt. Rewrite it in your own
# words — the agent is only as useful as this paragraph is specific.
INTEREST_PROFILE: str = """\
I build AI agents in Python and teach other developers how to do it.

I care a lot about:
- Agent architecture: planning, memory, tool use, human-in-the-loop, multi-agent
- Things I can run for free or locally: small models, open weights, free API tiers
- Concrete engineering writeups with code, benchmarks, or postmortems
- New model or framework releases that change how agents get built

I do not care about:
- Funding rounds, executive moves, company politics, valuations
- Policy, regulation, AI safety position pieces
- Consumer app launches and chatbot product news
- Pure theory with no runnable artifact
"""


# --- Stub mode. Set by the --stub flag; runs the whole night with canned model
# responses and zero Groq requests. Read at call time, so flipping it works.
STUB_LLM: bool = os.getenv("NIGHT_SCOUT_STUB", "") == "1"


def require_keys() -> None:
    """Fail early with a useful message instead of a stack trace mid-night."""
    if STUB_LLM:
        return
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and paste a free "
            "key from https://console.groq.com/keys"
        )


def make_llm(temperature: float = 0.0) -> ChatGroq:
    """Build the Groq chat model. Retries absorb the free tier's occasional 429."""
    require_keys()
    return ChatGroq(
        model=MODEL,
        api_key=GROQ_API_KEY,
        temperature=temperature,
        max_tokens=MAX_TOKENS,
        reasoning_effort=REASONING_EFFORT,
        max_retries=3,
    )
