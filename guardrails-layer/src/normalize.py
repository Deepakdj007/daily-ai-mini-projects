"""Layer 1: fold text into a form the other layers can actually read.

Inputs:  raw text from a user turn or a retrieved document
Outputs: cleaned text plus a list of what was stripped

This layer exists because the classifier downstream is a text model. Text it
cannot see, it cannot flag. Unicode Tag characters render as nothing at all in
a terminal or a browser, and full-width letters are a different codepoint from
the ASCII ones a pattern matcher looks for.
"""

from __future__ import annotations

import unicodedata

# Unicode Tag block. Copies of ASCII that render as zero pixels. A model
# reading the token stream sees the letters; a human reviewing the ticket
# sees an empty string.
TAG_START, TAG_END = 0xE0000, 0xE007F

ZERO_WIDTH = {
    "​",  # zero width space
    "‌",  # zero width non-joiner
    "‍",  # zero width joiner
    "﻿",  # byte order mark
    "­",  # soft hyphen
}

# Direction overrides can reorder how a sentence displays without changing
# the bytes a model consumes.
BIDI = {
    "‪", "‫", "‬", "‭", "‮",
    "⁦", "⁧", "⁨", "⁩",
}


def decode_tag_chars(text: str) -> str:
    """Return the ASCII hidden inside Unicode Tag characters, if any."""
    return "".join(
        chr(ord(ch) - TAG_START) for ch in text if TAG_START <= ord(ch) <= TAG_END
    )


def normalize(text: str) -> tuple[str, list[str]]:
    """Strip invisible characters and fold lookalikes to their ASCII form.

    Returns the cleaned text and a list of human-readable findings. The
    findings matter as much as the cleaning: text that needed stripping is
    itself a signal, because ordinary customers do not send Tag characters.
    """
    findings: list[str] = []

    hidden = decode_tag_chars(text)
    if hidden:
        findings.append(f"unicode tag characters carrying hidden text: {hidden!r}")

    out_chars = []
    stripped_zero_width = 0
    stripped_bidi = 0
    for ch in text:
        code = ord(ch)
        if TAG_START <= code <= TAG_END:
            continue  # already surfaced in `hidden`
        if ch in ZERO_WIDTH:
            stripped_zero_width += 1
            continue
        if ch in BIDI:
            stripped_bidi += 1
            continue
        out_chars.append(ch)

    if stripped_zero_width:
        findings.append(f"{stripped_zero_width} zero-width character(s) stripped")
    if stripped_bidi:
        findings.append(f"{stripped_bidi} bidi override character(s) stripped")

    cleaned = "".join(out_chars)

    # NFKC folds full-width and other compatibility forms onto plain ASCII,
    # so "Ｉｇｎｏｒｅ" and "Ignore" become the same string.
    folded = unicodedata.normalize("NFKC", cleaned)
    if folded != cleaned:
        findings.append("compatibility characters folded to ASCII by NFKC")

    # The hidden text is appended so the classifier downstream gets to read
    # it. Deleting it silently would hand the attacker a way to make text
    # invisible to the guard as well as to the reviewer.
    if hidden:
        folded = f"{folded}\n[recovered hidden text] {hidden}"

    return folded, findings
