"""Layer 2: score text with Meta Prompt Guard 2, in windows it can fit.

Inputs:  a block of text (a user turn, or one retrieved document)
Outputs: the highest injection probability found, and the per-window scores

Prompt Guard 2 has a 512-token context window and Groq rejects anything
longer with a 400 rather than truncating it. Sending a document straight in
therefore fails on the first realistic help article - and catching that error
and skipping the scan would hand an attacker the simplest bypass there is:
pad the injection with a thousand tokens of filler.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from src.config import GUARD_WINDOW_CHARS, GUARD_WINDOW_OVERLAP
from src.llm import guard_score


@dataclass
class ScanResult:
    """The outcome of scanning one block of text."""

    score: float
    windows: list[float] = field(default_factory=list)
    worst_window: str = ""

    @property
    def window_count(self) -> int:
        """How many separate calls the scan took."""
        return len(self.windows)


def split_windows(text: str) -> list[str]:
    """Cut text into overlapping windows small enough for a 512-token model.

    The overlap matters: an injection that straddles a clean cut would be
    split into two halves, each individually harmless-looking.
    """
    if len(text) <= GUARD_WINDOW_CHARS:
        return [text]

    step = GUARD_WINDOW_CHARS - GUARD_WINDOW_OVERLAP
    return [
        text[i : i + GUARD_WINDOW_CHARS]
        for i in range(0, len(text), step)
        if text[i : i + GUARD_WINDOW_CHARS].strip()
    ]


async def scan(text: str) -> ScanResult:
    """Score every window of the text and keep the worst one.

    The worst window wins, not the average. A single planted sentence inside
    a long legitimate document is exactly the attack, and averaging it
    against a thousand innocent characters is how you miss it.
    """
    text = text.strip()
    if not text:
        return ScanResult(score=0.0)

    windows = split_windows(text)
    scores = await asyncio.gather(*(guard_score(w) for w in windows))
    top = max(range(len(scores)), key=lambda i: scores[i])
    return ScanResult(
        score=scores[top], windows=list(scores), worst_window=windows[top]
    )
