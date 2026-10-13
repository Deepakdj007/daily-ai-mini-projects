"""Turn a WhatsApp export into a list of messages.

Inputs:  a .txt produced by Export Chat -> Without Media
Outputs: Message records, plus counts of what was skipped

One export comes from one phone, so it has one line format throughout - but
that format differs by operating system and by phone locale. Rather than
guessing, the prefix pattern below covers every known variant at once: the
invisible direction marks WhatsApp writes at the start of a line, iOS's
[brackets] against Android's trailing dash, `/ - .` date separators, two- or
four-digit years, optional seconds, and am/pm in its several spellings.

Ported from github.com/Pustur/whatsapp-chat-parser, which has had far more
locales thrown at it than this project ever will.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PREFIX_RE = re.compile(
    r"^[\u200e\u200f]*\[?"
    r"(\d{1,4}[-/.]\s?\d{1,4}[-/.]\s?\d{1,4})"     # 12/03/26, 2026-03-12
    r"[,.]?\s\D*?"                                  # comma, spaces, junk
    r"(\d{1,2}[.:]\d{1,2}(?:[.:]\d{1,2})?)"         # 9:14 or 9:14:07
    r"(?:\s([ap]\.?\s?m\.?))?"                      # pm, p.m., PM, or absent
    r"\]?(?:\s-|:)?\s",
    re.IGNORECASE,
)

# A sender name is short and single-line. Anything else that happens to carry
# a colon is a notice, not a person.
AUTHOR_RE = re.compile(r"^(?!\s)(.{1,64}?):\s(.*)$", re.S)

_SEP_RE = re.compile(r"[-/.]")


@dataclass(frozen=True, slots=True)
class Message:
    """One message: when it was sent, who sent it, and what it said."""

    when: datetime
    sender: str
    text: str


@dataclass(frozen=True, slots=True)
class ParseReport:
    """What the parser did, so the reader can sanity-check the result."""

    messages: int
    system_lines: int
    continuation_lines: int
    date_order: str
    senders: dict[str, int]


def sniff_date_order(dates: list[str]) -> str:
    """Work out whether an export writes day-first, month-first or year-first.

    Unresolvable from any single line: 03/04/26 is two valid dates. Decided
    across the whole file by finding a value that can only be a day.
    """
    triples = [
        tuple(int(part) for part in _SEP_RE.split(date.replace(" ", "")))
        for date in dates
    ]
    if not triples:
        return "day-first"
    if any(first > 31 for first, _, _ in triples):
        return "year-first"
    first_over_12 = any(first > 12 for first, _, _ in triples)
    second_over_12 = any(second > 12 for _, second, _ in triples)
    if first_over_12 and second_over_12:
        raise SystemExit(
            "This export has dates where both of the first two numbers exceed "
            "12, which no calendar produces. The file may be concatenated from "
            "two different phones.")
    if second_over_12:
        return "month-first"
    return "day-first"          # India's default, and the fallback when the
                                # sample never happens to pass the 12th


def _to_datetime(date: str, time: str, meridiem: str | None, order: str) -> datetime:
    """Build a datetime from the three captured pieces."""
    parts = [int(p) for p in _SEP_RE.split(date.replace(" ", ""))]
    if order == "year-first":
        year, month, day = parts
    elif order == "month-first":
        month, day, year = parts
    else:
        day, month, year = parts
    if year < 100:
        year += 2000
    clock = [int(p) for p in re.split(r"[.:]", time)]
    hour, minute = clock[0], clock[1]
    second = clock[2] if len(clock) > 2 else 0
    if meridiem:
        flat = meridiem.lower().replace(".", "").replace(" ", "")
        if flat == "pm" and hour != 12:
            hour += 12
        elif flat == "am" and hour == 12:
            hour = 0
    return datetime(year, month, day, hour, minute, second)


def _records(raw: str) -> tuple[list[tuple[re.Match[str], str]], int]:
    """Split the file into records, folding continuation lines into the one above.

    A line that does not start with a timestamp is the rest of the message
    before it. Miss this and every multi-line message silently becomes a
    truncated one, which is the most common way a chat parser goes wrong.
    """
    records: list[tuple[re.Match[str], str]] = []
    continuations = 0
    lines = raw.replace("\r\n", "\n").split("\n")
    if lines and not lines[-1]:
        lines.pop()     # the newline every text file ends with, which would
                        # otherwise land on the last message as a blank line
    for line in lines:
        match = PREFIX_RE.match(line)
        if match:
            records.append((match, line[match.end():]))
        elif records:
            head, body = records[-1]
            records[-1] = (head, f"{body}\n{line}")
            continuations += 1
    return records, continuations


def parse_export(path: Path) -> tuple[list[Message], ParseReport]:
    """Read an export and return its messages plus a report on the parse."""
    raw = path.read_text(encoding="utf-8-sig")       # exports can carry a BOM
    records, continuations = _records(raw)
    order = sniff_date_order([head.group(1) for head, _ in records])

    messages: list[Message] = []
    system = 0
    senders: dict[str, int] = {}
    for head, body in records:
        author = AUTHOR_RE.match(body.strip("\u200e\u200f"))
        if not author:      # a timestamped line with no sender is a notice:
            system += 1     # encryption, someone joining, a changed subject
            continue
        sender, text = author.group(1).strip(), author.group(2).strip()
        messages.append(_message(head, sender, text, order))
        senders[sender] = senders.get(sender, 0) + 1

    report = ParseReport(len(messages), system, continuations, order,
                         dict(sorted(senders.items(), key=lambda kv: -kv[1])))
    return messages, report


def _message(head: re.Match[str], sender: str, text: str, order: str) -> Message:
    """Assemble one Message from a matched prefix and its split body."""
    when = _to_datetime(head.group(1), head.group(2), head.group(3), order)
    return Message(when=when, sender=sender, text=text)


if __name__ == "__main__":
    import tempfile

    # Both operating systems, a narrow no-break space before "pm" (iOS writes
    # U+202F there), a leading direction mark, a two-line message, and two
    # notices that must not become messages.
    fixture = (
        "\ufeff12/03/26, 9:14 pm - Messages and calls are end-to-end encrypted.\n"
        "12/03/26, 9:15 pm - Arjun Mehta: arre bhai\n"
        "12/03/26, 9:16 pm - Nikhil: Can you send the deck?\n"
        "12/03/26, 9:17 pm - Arjun Mehta: haan one sec\n"
        "still uploading\n"
        "13/03/26, 10:05 am - Arjun Mehta: <Media omitted>\n"
    )
    ios = (
        "\u200e[12/03/26, 9:14:07\u202fpm] Nikhil: Morning\n"
        "\u200e[13/03/26, 9:15:00\u202fam] Arjun Mehta: subah subah kya hua\n"
    )
    checks: list[tuple[str, bool]] = []

    with tempfile.TemporaryDirectory() as tmp:
        android_path = Path(tmp) / "android.txt"
        android_path.write_text(fixture, encoding="utf-8")
        msgs, rep = parse_export(android_path)
        checks += [
            ("android prefix parses", rep.messages == 4),
            ("BOM does not break line 1", rep.system_lines == 1),
            ("notice is not a message", "Messages" not in {m.sender for m in msgs}),
            ("continuation folded in", msgs[2].text == "haan one sec\nstill uploading"),
            ("continuation counted", rep.continuation_lines == 1),
            ("pm becomes 21:00", msgs[0].when.hour == 21),
            ("am stays 10:00", msgs[3].when.hour == 10),
            ("sender counted", rep.senders["Arjun Mehta"] == 3),
        ]

        ios_path = Path(tmp) / "ios.txt"
        ios_path.write_text(ios, encoding="utf-8")
        ios_msgs, ios_rep = parse_export(ios_path)
        checks += [
            ("ios brackets parse", ios_rep.messages == 2),
            ("U+202F before pm", ios_msgs[0].when.hour == 21),
            ("U+200E stripped", ios_msgs[1].sender == "Arjun Mehta"),
            ("seconds kept", ios_msgs[0].when.second == 7),
        ]

    checks += [
        ("day-first when a day exceeds 12", sniff_date_order(["13/03/26"]) == "day-first"),
        ("month-first when the second does", sniff_date_order(["03/13/26"]) == "month-first"),
        ("year-first for ISO", sniff_date_order(["2026-03-12"]) == "year-first"),
        ("ambiguous falls back to day-first", sniff_date_order(["03/04/26"]) == "day-first"),
    ]

    for name, ok in checks:
        print(f"{'ok  ' if ok else 'FAIL'}  {name}")
    raise SystemExit(0 if all(ok for _, ok in checks) else 1)
