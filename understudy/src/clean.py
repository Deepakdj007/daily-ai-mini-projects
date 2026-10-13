"""Throw away what is not voice, and redact what should never leave the laptop.

Inputs:  Messages from parse.py, then Pairs from sessionize.py
Outputs: the same, minus noise, with personal details replaced

Cleaning runs on messages *before* they are grouped into turns. Drop a photo
after pairing and you leave a hole in the middle of a conversation; drop it
first and the turns either side merge the way they should.

Two rules here are not obvious.

Links are replaced, not deleted. Sending links is part of how a person talks,
so the model should learn that you send them - just not which ones.

One-word replies stay. "haan", "ok da" and a thumbs-up carry more of a voice
than any paragraph does. They are capped at a share of the dataset rather
than removed, because a model trained mostly on them learns to say nothing.
"""

from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass

from src.config import (MAX_REPLY_CHARS, MIN_REPLY_CHARS, RANDOM_SEED,
                        SHORT_REPLY_CHARS, SHORT_REPLY_MAX_SHARE)
from src.parse import Message
from src.sessionize import Pair

# WhatsApp writes these in the phone's own language, so this list is English
# only and will miss a Hindi or Tamil export. That is a real limit, not an
# oversight: the reply itself is dropped for being empty afterwards anyway.
NOISE_RE = re.compile(
    r"^(?:<(?:media|attachment) omitted>|(?:image|video|audio|sticker|gif|"
    r"document|contact card|voice call|video call) omitted|"
    r"this message was deleted|you deleted this message|null|"
    r"missed voice call|missed video call)\.?$",
    re.IGNORECASE)

EDITED_RE = re.compile(r"\s*<this message was edited>\s*$", re.IGNORECASE)
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

# Redaction runs before anything is written, because the file it produces is
# uploaded to a GPU service. UPI handles are matched before e-mail so that
# arjun@okaxis is not mistaken for an address.
REDACTIONS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("upi", re.compile(r"\b[\w.\-]{3,}@(?:okaxis|oksbi|okhdfcbank|okicici|ybl|"
                       r"paytm|upi|ibl|axl|apl|axisb)\b", re.IGNORECASE)),
    ("email", re.compile(r"\b[\w.+\-]+@[\w\-]+\.[\w.\-]+\b")),
    # Nobody writes ten digits in a row: 98765 43210 and +91-98765-43210 are
    # how the same number actually appears in a chat.
    ("phone", re.compile(r"(?:\+?91[\s\-]?)?\b[6-9]\d{4}[\s\-]?\d{5}\b")),
    ("number", re.compile(r"\b\d{6,}\b")),      # OTPs, account and order numbers
)


@dataclass(slots=True)
class CleanReport:
    """What cleaning removed, so none of it is silent."""

    kept: int
    noise_dropped: int
    links: int
    redactions: dict[str, int]


@dataclass(slots=True)
class FilterReport:
    """What pair filtering removed, for the same reason."""

    kept: int
    too_long: int
    too_short: int
    duplicates: int
    short_downsampled: int


def scrub(text: str) -> tuple[str, dict[str, int], int]:
    """Replace links and personal details. Returns the text and what it hit."""
    counts: dict[str, int] = {}
    text, links = URL_RE.subn("<link>", text)
    for name, pattern in REDACTIONS:
        text, hits = pattern.subn(f"<{name}>", text)
        if hits:
            counts[name] = counts.get(name, 0) + hits
    return text, counts, links


def clean_messages(messages: list[Message]) -> tuple[list[Message], CleanReport]:
    """Drop the messages that carry no words, and redact the ones that do."""
    kept: list[Message] = []
    noise = 0
    links = 0
    redactions: dict[str, int] = {}
    for message in messages:
        text = EDITED_RE.sub("", message.text).strip()
        if not text or NOISE_RE.match(text):
            noise += 1
            continue
        text, counts, hits = scrub(text)
        links += hits
        for name, count in counts.items():
            redactions[name] = redactions.get(name, 0) + count
        text = text.strip()
        if not text:
            noise += 1
            continue
        kept.append(Message(message.when, message.sender, text))
    return kept, CleanReport(len(kept), noise, links, redactions)


def _normalise(text: str) -> str:
    """Fold a reply to its bare words, for comparing two of them."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"[^\w\s]", "", folded).strip()


def filter_pairs(pairs: list[Pair]) -> tuple[list[Pair], FilterReport]:
    """Apply the length caps, drop repeats, and cap the share of one-worders."""
    kept: list[Pair] = []
    too_long = too_short = duplicates = 0
    seen: set[str] = set()
    for pair in pairs:
        length = len(pair.reply)
        if length > MAX_REPLY_CHARS:
            too_long += 1
            continue
        if length < MIN_REPLY_CHARS:
            too_short += 1
            continue
        # Only long replies are deduplicated. "ok" recurring forty times is
        # not a duplicate, it is the shape of the person.
        if length > SHORT_REPLY_CHARS:
            key = _normalise(pair.reply)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
        kept.append(pair)

    short = [p for p in kept if len(p.reply) <= SHORT_REPLY_CHARS]
    long = [p for p in kept if len(p.reply) > SHORT_REPLY_CHARS]
    allowance = int(len(long) * SHORT_REPLY_MAX_SHARE / (1 - SHORT_REPLY_MAX_SHARE))
    downsampled = 0
    if len(short) > allowance:
        random.Random(RANDOM_SEED).shuffle(short)
        downsampled = len(short) - allowance
        short = short[:allowance]
        keep_ids = {p.pair_id for p in short} | {p.pair_id for p in long}
        kept = [p for p in kept if p.pair_id in keep_ids]

    return kept, FilterReport(len(kept), too_long, too_short, duplicates,
                              downsampled)


if __name__ == "__main__":
    from datetime import datetime

    from src.sessionize import Turn

    def msg(text: str, sender: str = "Arjun") -> Message:
        return Message(datetime(2026, 3, 1, 9, 0), sender, text)

    raw = [
        msg("<Media omitted>"),
        msg("This message was deleted"),
        msg("bhej diya <This message was edited>"),
        msg("check https://example.com/x bhai"),
        msg("call me on +91 98765 43210"),
        msg("arjun@okaxis se bhej"),
        msg("mail me at arjun@gmail.com"),
        msg("otp is 482913"),
        msg("null"),
    ]
    cleaned, report = clean_messages(raw)
    texts = [m.text for m in cleaned]

    def pair(reply: str, pid: str) -> Pair:
        return Pair(pid, [Turn(False, "kya")], reply)

    many = ([pair("ok", f"s{i}") for i in range(50)]
            + [pair(f"a longer reply number {i}", f"l{i}") for i in range(20)])
    filtered, freport = filter_pairs(many)
    short_share = sum(1 for p in filtered if len(p.reply) <= SHORT_REPLY_CHARS) / len(filtered)

    dupes, dreport = filter_pairs([pair("same long reply here", "a"),
                                   pair("Same long reply here!", "b")])

    checks = [
        ("media dropped", "<Media omitted>" not in texts),
        ("deleted dropped", not any("deleted" in t for t in texts)),
        ("noise counted", report.noise_dropped == 3),
        ("edited marker stripped", "bhej diya" in texts),
        ("link replaced not deleted", "check <link> bhai" in texts),
        ("link counted", report.links == 1),
        ("phone redacted", "call me on <phone>" in texts),
        ("upi redacted", "<upi> se bhej" in texts),
        ("email redacted", "mail me at <email>" in texts),
        ("otp redacted", "otp is <number>" in texts),
        ("no digits survive", not any(re.search(r"\d{6,}", t) for t in texts)),
        ("short replies capped", short_share <= SHORT_REPLY_MAX_SHARE + 0.001),
        ("cap is reported", freport.short_downsampled > 0),
        ("long duplicate dropped", dreport.duplicates == 1 and len(dupes) == 1),
    ]
    for name, ok in checks:
        print(f"{'ok  ' if ok else 'FAIL'}  {name}")
    raise SystemExit(0 if all(ok for _, ok in checks) else 1)
