"""Group messages into conversations and turn them into training pairs.

Inputs:  Messages from parse.py, and whose replies to target
Outputs: Pair records - some context, then one reply by the target speaker

Two things happen here that decide how the fine-tune turns out.

A conversation is split on a six-hour gap. Thirty minutes cuts a thread in
half every time someone answers after a meeting; a day glues last night onto
this morning.

Consecutive messages by one person are merged into a single turn. People send
four short messages in a row, and those four are one reply. Train on them
separately and you teach the model to answer in fragments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

from src.config import CONTEXT_TURNS, GAP_HOURS, MAX_CONTEXT_CHARS
from src.parse import Message


@dataclass(frozen=True, slots=True)
class Turn:
    """One side speaking once, however many bubbles that took."""

    is_target: bool
    text: str


@dataclass(slots=True)
class Pair:
    """What the model sees, and what it should have said."""

    pair_id: str
    context: list[Turn]
    reply: str
    session: int = 0
    extras: dict[str, str] = field(default_factory=dict)

    def as_chat(self) -> dict[str, list[dict[str, str]]]:
        """Render to the messages format every chat template expects."""
        turns = [{"role": "assistant" if turn.is_target else "user",
                  "content": turn.text} for turn in self.context]
        turns.append({"role": "assistant", "content": self.reply})
        return {"messages": turns}


def split_sessions(messages: list[Message]) -> list[list[Message]]:
    """Cut the message stream wherever nobody spoke for GAP_HOURS."""
    if not messages:
        return []
    gap = timedelta(hours=GAP_HOURS)
    sessions: list[list[Message]] = [[messages[0]]]
    for previous, current in zip(messages, messages[1:]):
        if current.when - previous.when > gap:
            sessions.append([])
        sessions[-1].append(current)
    return sessions


def merge_turns(session: list[Message], target: str, label_senders: bool) -> list[Turn]:
    """Collapse runs into turns that strictly alternate target / not-target.

    Alternation is not cosmetic: a chat template that receives two user turns
    in a row produces a prompt no instruct model was trained on. In a group
    chat the other side is several people, so their names go inline - that is
    the only way the model can tell who it is answering.
    """
    turns: list[Turn] = []
    for message in session:
        is_target = message.sender == target
        text = message.text
        if not is_target and label_senders:
            text = f"{message.sender}: {text}"
        if turns and turns[-1].is_target == is_target:
            turns[-1] = Turn(is_target, f"{turns[-1].text}\n{text}")
        else:
            turns.append(Turn(is_target, text))
    return turns


def _trim(context: list[Turn]) -> list[Turn]:
    """Drop the oldest context turns until the context fits the budget."""
    kept = list(context)
    while kept and sum(len(turn.text) for turn in kept) > MAX_CONTEXT_CHARS:
        kept.pop(0)
    return kept


def build_pairs(messages: list[Message], target: str,
                label_senders: bool = False, prefix: str = "p") -> list[Pair]:
    """Every time the target speaks after somebody else, that is one example."""
    pairs: list[Pair] = []
    for index, session in enumerate(split_sessions(messages)):
        turns = merge_turns(session, target, label_senders)
        for position, turn in enumerate(turns):
            if not turn.is_target or position == 0:
                continue        # a session the target opened has no context,
                                # so there is nothing to learn a reply from
            context = _trim(turns[max(0, position - CONTEXT_TURNS):position])
            if not context:
                continue
            pairs.append(Pair(f"{prefix}{len(pairs):04d}", context,
                              turn.text, session=index))
    return pairs


def is_group_chat(senders: dict[str, int]) -> bool:
    """More than two people means the other side needs naming."""
    return len(senders) > 2


def split_heldout(pairs: list[Pair], wanted: int) -> tuple[list[Pair], list[Pair]]:
    """Hold out whole recent conversations, never individual pairs.

    Splitting pair by pair would put one half of a conversation in training
    and the other half in the test set, and the model would be scored on
    replies whose neighbours it had already memorised.
    """
    if not pairs:
        return [], []
    # Never hand more than a quarter of a small dataset to testing - on a
    # short export the recent-session rule alone can hold out most of it.
    ceiling = min(wanted, max(1, len(pairs) // 4))
    sessions = sorted({pair.session for pair in pairs}, reverse=True)
    held: set[int] = set()
    count = 0
    for session in sessions:
        if count >= ceiling or len(held) == len(sessions) - 1:
            break
        held.add(session)
        count += sum(1 for pair in pairs if pair.session == session)
    return ([p for p in pairs if p.session not in held],
            [p for p in pairs if p.session in held])


if __name__ == "__main__":
    from datetime import datetime

    def msg(hour: int, sender: str, text: str, day: int = 1) -> Message:
        return Message(datetime(2026, 3, day, hour, 0), sender, text)

    burst = [
        msg(9, "Nikhil", "you up?"),
        msg(9, "Arjun", "haan"),
        msg(9, "Arjun", "bol"),
        msg(9, "Arjun", "kya hua"),
        msg(10, "Nikhil", "deck ready?"),
        msg(10, "Arjun", "almost"),
        msg(20, "Nikhil", "and now?"),          # 10 hours later - new session
        msg(20, "Arjun", "done bhai"),
    ]
    pairs = build_pairs(burst, "Arjun")
    sessions = split_sessions(burst)

    opener = [msg(9, "Arjun", "good morning"), msg(9, "Nikhil", "morning")]

    group = [
        msg(9, "Sneha", "dinner?"),
        msg(9, "Ravi", "where"),
        msg(9, "Arjun", "koramangala"),
    ]
    group_pairs = build_pairs(group, "Arjun", label_senders=True)

    long_ctx = [msg(9, "Nikhil", "x" * 900), msg(9, "Nikhil", "y" * 900),
                msg(9, "Arjun", "ok")]
    # Two 900-char turns by one sender merge into one 1801-char turn, which
    # alone busts the 1200 budget - trimming it leaves nothing to answer.
    trimmed = build_pairs(long_ctx, "Arjun")

    spread = [Pair(f"h{i}", [Turn(False, "q")], "a", session=i // 2)
              for i in range(40)]
    train, held = split_heldout(spread, 4)
    small = [Pair(f"s{i}", [Turn(False, "q")], "a", session=i // 2)
             for i in range(10)]
    _, capped = split_heldout(small, 999)

    checks = [
        ("gap splits sessions", len(sessions) == 2),
        ("held out by whole session", len(held) == 4 and len(train) == 36),
        ("held out sessions are recent",
            {p.session for p in held} == {18, 19}),
        ("no session spans the split",
            not ({p.session for p in held} & {p.session for p in train})),
        ("held out is capped at a quarter", len(capped) == 2),
        ("training is never emptied", split_heldout(small, 999)[0] != []),
        ("burst merges into one turn", pairs[0].reply == "haan\nbol\nkya hua"),
        ("pairs found in both sessions", len(pairs) == 3),
        ("context is the other side", pairs[0].context[-1].text == "you up?"),
        ("roles alternate", [t["role"] for t in pairs[1].as_chat()["messages"]]
            == ["user", "assistant", "user", "assistant"]),
        ("session opener is skipped", build_pairs(opener, "Arjun") == []),
        ("group names the speakers",
            group_pairs[0].context[0].text == "Sneha: dinner?\nRavi: where"),
        ("oversized context drops the pair", trimmed == []),
    ]
    for name, ok in checks:
        print(f"{'ok  ' if ok else 'FAIL'}  {name}")
    raise SystemExit(0 if all(ok for _, ok in checks) else 1)
