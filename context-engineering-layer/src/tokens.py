"""Token counting, message elision, and budget splitting.

Inputs:  strings, Messages and Turns
Outputs: token counts, truncated text, and (kept, dropped) turn splits

Imports nothing from config.py, because every layer depends on this module and
the layer has to be liftable. The tokenizer fudge factor is passed in, not read
from the environment.

The count here is a local estimate, not Groq's. tiktoken prices raw text;
the Harmony chat template adds role headers and per-message framing that the
raw encoding never sees. `main.py cache-probe` measures the real ratio against
usage.prompt_tokens and stores it as TOKENIZER_FUDGE - every budget assertion
in the project is only as honest as that calibration.
"""

from __future__ import annotations

import functools
from typing import Sequence

import tiktoken

from src.types import Message, Turn

ENCODING_NAME = "o200k_harmony"

# Rough per-message overhead in the chat template: role header, separators and
# the end-of-message token. Calibration refines the whole estimate; this just
# stops short messages being wildly under-counted before calibration runs.
PER_MESSAGE_OVERHEAD = 4

ELISION = "\n[... {n} tokens elided ...]\n"


@functools.lru_cache(maxsize=1)
def encoding() -> tiktoken.Encoding:
    """The gpt-oss tokenizer, loaded once.

    First call downloads and caches the BPE file, so it needs the network once.
    """
    return tiktoken.get_encoding(ENCODING_NAME)


def count_text(text: str, fudge: float = 1.0) -> int:
    """Token count for a bare string."""
    return int(len(encoding().encode(text, disallowed_special=())) * fudge)


def count_message(msg: Message, fudge: float = 1.0) -> int:
    """Token count for one message, including chat-template overhead."""
    body = msg.content if msg.role != "tool" else f"[tool:{msg.name}]\n{msg.content}"
    return count_text(body, fudge) + PER_MESSAGE_OVERHEAD


def count_messages(messages: Sequence[Message], fudge: float = 1.0) -> int:
    """Token count for a message sequence."""
    return sum(count_message(m, fudge) for m in messages)


def count_turn(turn: Turn, fudge: float = 1.0) -> int:
    """Token count for one whole turn."""
    return count_messages(turn.messages, fudge)


def elide_middle(text: str, *, head: int, tail: int, fudge: float = 1.0) -> str:
    """Keep the first `head` and last `tail` tokens, elide the rest.

    A pure function of the text: no budget, no position, no conversation
    length. That is what makes capping prefix-stable - an old message renders
    byte-identically on turn 40 as it did on turn 4, so the provider's prefix
    cache still hits. Any cap that varied with remaining budget would rewrite
    history every turn and forfeit the cache entirely.

    The marker states the elided count so the model can say a field was cut
    rather than silently inventing it.
    """
    enc = encoding()
    ids = enc.encode(text, disallowed_special=())
    marker_cost = count_text(ELISION.format(n=0), fudge) + 2
    if len(ids) <= head + tail + marker_cost:
        return text
    dropped = len(ids) - head - tail
    return (
        enc.decode(ids[:head])
        + ELISION.format(n=dropped)
        + enc.decode(ids[len(ids) - tail :])
    )


def split_by_budget(
    turns: Sequence[Turn],
    budget: int,
    *,
    min_turns: int = 1,
    fudge: float = 1.0,
) -> tuple[tuple[Turn, ...], tuple[Turn, ...]]:
    """Split turns into (kept, dropped), keeping the most recent that fit.

    Returns kept in chronological order. `min_turns` guarantees the live
    exchange survives even when a single turn is larger than the whole budget -
    without it a fat tool result could empty the context completely.
    """
    kept: list[Turn] = []
    spent = 0
    for turn in reversed(turns):
        cost = count_turn(turn, fudge)
        if kept and spent + cost > budget and len(kept) >= min_turns:
            break
        if not kept and cost > budget and min_turns < 1:
            break
        kept.append(turn)
        spent += cost
    kept.reverse()
    keep_ids = {t.index for t in kept}
    dropped = tuple(t for t in turns if t.index not in keep_ids)
    return tuple(kept), dropped


def common_prefix_tokens(a: str, b: str) -> int:
    """Length of the longest shared token prefix of two rendered prompts.

    This is the whole cache measurement, and it costs nothing. Groq caches a
    prefix, so this number is exactly how much of a turn could be served from
    cache - exact rather than sampled, and computable for every arm offline
    instead of for the one arm a live placement experiment could afford.
    """
    enc = encoding()
    ta = enc.encode(a, disallowed_special=())
    tb = enc.encode(b, disallowed_special=())
    n = 0
    for x, y in zip(ta, tb):
        if x != y:
            break
        n += 1
    return n


if __name__ == "__main__":
    blob = '{"merchant": "M-88213", "window_s": 47, "rows": [' + '{"a":1},' * 400 + "]}"
    full = count_text(blob)
    capped = elide_middle(blob, head=70, tail=30)
    print(f"tool blob {full} tok -> capped {count_text(capped)} tok")
    assert count_text(capped) < full
    assert "elided" in capped
    assert blob[:20] in capped and blob[-15:] in capped, "head and tail must survive"

    short = "already small"
    assert elide_middle(short, head=70, tail=30) == short, "short text is untouched"

    turns = tuple(
        Turn(i, (Message("user", f"question {i} " + "pad " * 40),
                 Message("assistant", f"reply {i} " + "pad " * 40)))
        for i in range(10)
    )
    per = count_turn(turns[0])
    kept, dropped = split_by_budget(turns, per * 3 + 5)
    print(f"turn={per} tok, budget={per * 3 + 5} -> kept {[t.index for t in kept]}")
    assert len(kept) == 3 and kept[-1].index == 9, "must keep the most recent"
    assert len(dropped) == 7 and dropped[0].index == 0

    tiny_kept, _ = split_by_budget(turns, 1)
    assert len(tiny_kept) == 1, "min_turns must protect the live exchange"

    a = "system prompt\nturn one\nturn two\nquestion A"
    b = "system prompt\nturn one\nturn two\nquestion B"
    c = "system prompt\nturn two\nturn three\nquestion A"
    print(f"append-only shares {common_prefix_tokens(a, b)} tok; "
          f"after eviction {common_prefix_tokens(a, c)} tok")
    assert common_prefix_tokens(a, b) > common_prefix_tokens(a, c), (
        "an evicting window must share less prefix than an appending one"
    )
    print("token accounting holds")
