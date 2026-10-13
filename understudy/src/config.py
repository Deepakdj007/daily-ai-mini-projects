"""Every knob in one place.

Inputs:  environment variables from .env
Outputs: constants, paths, resolve_me()

This project talks to no API and needs no key. The only thing it has to be
told is which participant in the export is you, because WhatsApp writes your
saved display name into the file rather than the word "You".
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUT_DIR = ROOT_DIR / "out"
SAMPLE_CHAT = DATA_DIR / "sample_chat.txt"
FACTS_PATH = DATA_DIR / "facts.json"
# A real Kaggle run, committed so `score` works before you have run one
# yourself. Replace it with your own after training.
RESPONSES_PATH = DATA_DIR / "responses.json"
GATED_FACTS_PATH = OUT_DIR / "facts.json"     # only the questions that passed
TRAIN_PATH = OUT_DIR / "train.jsonl"
HELDOUT_PATH = OUT_DIR / "heldout.jsonl"
STRANGER_PATH = OUT_DIR / "stranger.jsonl"

load_dotenv(ROOT_DIR / ".env")

for _stream in (sys.stdout, sys.stderr):
    try:            # A WhatsApp export is full of emoji and Windows consoles
        _stream.reconfigure(encoding="utf-8", errors="replace")   # are cp1252
    except (AttributeError, ValueError):
        pass

ME = os.getenv("ME", "").strip()

# Six hours. Thirty minutes splits a conversation every time someone replies
# after a meeting; twenty-four hours glues yesterday onto this morning.
GAP_HOURS = 6.0

# How many previous turns ride along as context. Three is enough to resolve
# "he"/"that one" without pushing sequence length up on a free T4.
CONTEXT_TURNS = 3

MIN_REPLY_CHARS = 1
MAX_REPLY_CHARS = 300
MAX_CONTEXT_CHARS = 1200

# "haan", "ok da" and a thumbs-up are the highest-signal style tokens a person
# has, so they stay - but a dataset that is half one-word replies teaches the
# model to say nothing. Cap the share instead of dropping the class.
SHORT_REPLY_CHARS = 4
SHORT_REPLY_MAX_SHARE = 0.15

HELDOUT_PAIRS = 40          # per speaker: yours, and the stranger control's
RANDOM_SEED = 3407


def resolve_me(senders: list[str], override: str | None = None) -> str:
    """Decide which participant is the reader, or fail with the real options.

    Order: an explicit --me flag, then ME from .env. Never guessed - an export
    gives no structural hint about which name belongs to the phone's owner.
    """
    choice = (override or ME).strip()
    if not choice:
        raise SystemExit(
            "Nobody is set as you. Run `inspect` to see the senders in your "
            "export, then either pass --me \"Your Name\" or put ME=\"Your Name\" "
            "in .env")
    if choice not in senders:
        listed = "\n  ".join(senders) or "(none found)"
        raise SystemExit(
            f"{choice!r} does not appear in this export. The senders are:\n  "
            f"{listed}\nThe name has to match exactly, spaces and all.")
    return choice


def ensure_dirs() -> None:
    """Create the directories the pipeline writes into."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print(f"me            {ME or 'NOT SET - pass --me or set ME in .env'}")
    print(f"session gap   {GAP_HOURS} hours")
    print(f"context       {CONTEXT_TURNS} turns, max {MAX_CONTEXT_CHARS} chars")
    print(f"reply length  {MIN_REPLY_CHARS}-{MAX_REPLY_CHARS} chars")
    print(f"short replies capped at {SHORT_REPLY_MAX_SHARE:.0%} of the dataset")
    print(f"held out      {HELDOUT_PAIRS} pairs each for you and the control")
    print(f"sample chat   {'found' if SAMPLE_CHAT.exists() else 'MISSING'}")
