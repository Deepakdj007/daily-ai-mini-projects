"""Expand the fact spec into a transcript, derive the zones, run every gate.

Inputs:  data/scenario.json
Outputs: 100 turns, 28 probes, the per-arm window-reach table, and a pass/fail
         on the build gates - all at zero API cost

The transcript is generated, never hand-authored. Twenty-eight probes with
carriers, zones and entropy floors cannot be kept consistent by hand, and a
spec that expands deterministically is the only version a reader can reproduce.

Zone boundaries are DERIVED from the measured window reach of two reference
arms, not hardcoded. Hardcoding them is how a zone silently drifts out of sync
with the arm it is supposed to describe, and then the position axis is a
function of the treatment rather than of position.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
from dataclasses import replace
from typing import Any

from src import config, tokens
from src.layers.base import PipelineConfig
from src.gates import gates as run_gates
from src.types import Carrier, Fact, Message, Turn, Zone

FILLER_USER = [
    "any movement on that?", "what does the desk show now?", "can you pull the latest?",
    "is that still climbing?", "how does that compare to before?", "anything on the edge?",
    "and the downstream side?", "did that clear?", "what about the other region?",
    "give me the current read", "is the queue draining?", "any change after the push?",
]
FILLER_ASSISTANT = [
    "Holding steady for now, nothing new on the board.",
    "Numbers are flat against the last read.",
    "Pulled it - no change worth flagging.",
    "Still within the usual band.",
    "Queue is moving, nothing stuck.",
    "Downstream looks unaffected so far.",
]
# Filler has to read like the conversation it is padding. An earlier version
# padded with the word "detail" repeated, and the model answered UNKNOWN even
# when the planted fact was sitting in the context - the surrounding text
# looked corrupted, so it declined to read any of it. That is a transcript bug
# that presents as a memory result, which is the most expensive kind.
_FILLER_CLAUSES = [
    "Nothing else jumped out on that pass.",
    "Graphs look the same as the previous window.",
    "I will keep an eye on it for another few minutes.",
    "No alerts fired off the back of that.",
    "The other regions are quiet so far.",
    "Worth a second look if it moves again.",
    "That matches what the dashboard is showing.",
    "No customer reports tied to it yet.",
    "Logging it here so the handover has it.",
    "Rechecked after the last deploy and it held.",
    "Queue depth is where we would expect it.",
    "Nothing pending on the approval side.",
]


def _pad_to(text: str, target: int, rng: random.Random | None = None) -> str:
    """Pad with realistic desk chatter until the text costs about `target`."""
    picker = rng or random.Random(len(text))
    guard = 0
    while tokens.count_text(text) < target and guard < 40:
        text += " " + picker.choice(_FILLER_CLAUSES)
        guard += 1
    return text.strip()


def load_spec() -> dict[str, Any]:
    """Read the fact spec."""
    return json.loads(config.SCENARIO_PATH.read_text(encoding="utf-8"))


def spec_hash() -> str:
    """Content hash of the spec, stamped onto every result.

    Without it a leaderboard cannot be traced back to the exact questions that
    produced it, and a quiet edit to one probe changes a number with no record.
    """
    blob = config.SCENARIO_PATH.read_bytes()
    return "sha256:" + hashlib.sha256(blob).hexdigest()[:12]


def _tool_body(schema: str, fields: list[str], rng: random.Random,
               planted: tuple[str, str] | None, carrier: Carrier,
               reserved: frozenset[str] = frozenset()) -> str:
    """One synthetic tool result, with a fact planted at head or in the middle.

    `reserved` holds every gold value. Filler is redrawn until it avoids them,
    because a random field that happens to mint the same value as a planted
    fact breaks the exactly-once invariant the whole scorecard rests on - and
    it breaks it silently, in a way that reads as a probe being answerable
    from two places rather than as a generator bug.
    """
    from src.grader import normalize

    banned = {normalize(v) for v in reserved}

    def draw(make: Any) -> str:
        for _ in range(50):
            candidate = str(make())
            if normalize(candidate) not in banned:
                return candidate
        return str(make())

    values: dict[str, str] = {}
    for name in fields:
        if name.endswith("_at"):
            values[name] = draw(lambda: f"{rng.randrange(0, 24):02d}:{rng.randrange(0, 60):02d}")
        elif name.endswith("_id"):
            prefix = {"batch": "SB", "incident": "INC", "merchant": "M"}[name.split("_")[0]]
            values[name] = draw(lambda: f"{prefix}-{rng.randrange(10000, 99999)}")
        elif name in {"reconciled"}:
            values[name] = rng.choice(["true", "false"])
        elif name in {"severity"}:
            values[name] = rng.choice(["sev1", "sev2", "sev3"])
        elif name in {"plan"}:
            values[name] = draw(lambda: rng.choice(["Starter", "Growth", "Scale"]))
        elif name in {"region", "shard"}:
            values[name] = rng.choice(["ap-south-1", "ap-south-2", "shard-04", "shard-11"])
        elif name in {"webhook_url"}:
            values[name] = draw(lambda: f"https://hooks-{rng.randrange(1000, 9999)}.example.in/psetu")
        elif name in {"contact", "acked_by"}:
            values[name] = rng.choice(["ops-desk", "sre-oncall", "billing-1"])
        elif name in {"error_ratio"}:
            values[name] = draw(lambda: f"0.{rng.randrange(1000, 9999)}")
        else:
            values[name] = draw(lambda: rng.randrange(100, 999999))

    if planted is not None:
        field_name, value = planted
        if field_name == "webhook_url":
            values[field_name] = f"https://{value}.example.in/psetu"
        else:
            values[field_name] = value
        if carrier == "tool_head":
            ordered = [field_name] + [f for f in fields if f != field_name]
        else:
            # Bury it past the head the cap keeps, so a tool_body fact is
            # genuinely in the elided middle rather than nominally so.
            rest = [f for f in fields if f != field_name]
            cut = max(2, len(rest) // 2)
            ordered = rest[:cut] + [field_name] + rest[cut:]
    else:
        ordered = list(fields)

    body = "{" + ", ".join(f'"{k}": "{values[k]}"' for k in ordered)
    body = _pad_to(body + ', "notes": "', config.TURN_TOOL_TOKENS - 2, rng) + '"}'
    return body


def build() -> tuple[tuple[Turn, ...], list[Fact], dict[str, Any]]:
    """Expand the spec into turns and probes."""
    spec = load_spec()
    rng = random.Random(20260828)
    by_turn = {f["turn"]: f for f in spec["facts"]}
    distractors = {d["turn"]: d for d in spec["distractors"]}
    schemas = spec["tool_schemas"]
    schema_names = list(schemas)

    reserved = frozenset(
        [f["value"] for f in spec["facts"]]
        + [d["value"] for d in spec["doc_probes"]]
        + [v for d in spec["distractors"] for v in d["overrides"].values()]
    )

    turns: list[Turn] = []
    for i in range(1, config.TRANSCRIPT_TURNS + 1):
        fact = by_turn.get(i)
        distractor = distractors.get(i)
        carrier: Carrier = fact["carrier"] if fact else "none"

        user = _pad_to(rng.choice(FILLER_USER), config.TURN_USER_TOKENS, rng)
        assistant = _pad_to(rng.choice(FILLER_ASSISTANT), config.TURN_ASSISTANT_TOKENS, rng)
        if fact:
            # A prose fact is asserted outright; a tool fact gets an anchor
            # that says WHICH record matters without naming the value. Without
            # one, a probe like "which partition carried the blame" is
            # unanswerable - dozens of turns carry a shard field and nothing
            # marks one of them - and the arm would score a miss that measures
            # the probe rather than the layer.
            framing = fact.get("statement") or fact.get("anchor") or ""
            assistant = _pad_to(framing, config.TURN_ASSISTANT_TOKENS, rng)

        schema = (fact or distractor or {}).get("schema") or rng.choice(schema_names)
        planted = (
            (fact["field"], fact["value"])
            if fact and carrier in ("tool_head", "tool_body")
            else None
        )
        body = _tool_body(schema, schemas[schema], rng, planted, carrier, reserved)
        if distractor:
            for k, v in distractor["overrides"].items():
                body = body.replace('"notes"', f'"{k}": "{v}", "notes"', 1)

        turns.append(Turn(i, (
            Message("user", user),
            Message("assistant", assistant),
            Message("tool", body, schema),
        )))

    facts = [
        Fact(id=f["id"], turn=f["turn"], zone="deep", carrier=f["carrier"],
             value=f["value"], probe=f["probe"], answer_space=f["answer_space"],
             schema=f.get("schema", ""), field_name=f.get("field", ""))
        for f in spec["facts"]
    ]
    facts += [
        Fact(id=n["id"], turn=-1, zone="negative", carrier="none", value="UNKNOWN",
             probe=n["probe"], answer_space=1000)
        for n in spec["negatives"]
    ]
    facts += [
        Fact(id=d["id"], turn=0, zone="doc", carrier="none", value=d["value"],
             probe=d["probe"], answer_space=1000, schema=d["source"])
        for d in spec["doc_probes"]
    ]
    return tuple(turns), facts, spec


async def window_reach_async(turns: tuple[Turn, ...]) -> dict[str, int]:
    """Oldest turn each reference arm's window can hold. Zero API cost."""
    from src.pipeline import assemble, make_deps

    deps = make_deps(fudge=config.TOKENIZER_FUDGE)
    base = PipelineConfig(context_budget=config.CONTEXT_BUDGET_TOKENS)
    arms = {
        "window": replace(base, cap=False, pin=False, retrieve=False, summarize=False),
        "cap+pin+window": replace(base, retrieve=False, summarize=False),
        "+summarize": replace(base, retrieve=False),
        "full": base,
    }
    spec = load_spec()
    out: dict[str, int] = {}
    for name, cfg in arms.items():
        ctx = await assemble(
            turns, "probe", cfg, deps,
            system_prompt=spec["system_prompt"], pins=spec["pinned_facts"],
            instruction=config.ANSWER_INSTRUCTION,
        )
        ids = ctx.window.turn_ids if ctx.window else ()
        out[name] = min(ids) if ids else config.TRANSCRIPT_TURNS + 1
    return out


def window_reach(turns: tuple[Turn, ...]) -> dict[str, int]:
    """Synchronous wrapper, for callers that are not already in a loop."""
    return asyncio.run(window_reach_async(turns))


def assign_zones(facts: list[Fact], reach: dict[str, int]) -> list[Fact]:
    """Label every fact by position, using the measured boundaries.

    `recent` is bounded by the NARROWEST window on the ladder, not the naive
    one. The full stack spends budget on a summary and a retrieved block, so
    its window reaches back less far than plain capping does - and a control
    probe that sat outside it would fail in the one arm the control exists to
    validate, voiding a run for a reason that is not a bug.
    """
    naive = max(reach.values())
    capped = min(reach[k] for k in ("cap+pin+window", "+summarize"))
    out: list[Fact] = []
    for f in facts:
        if f.zone in ("negative", "doc"):
            out.append(f)
            continue
        zone: Zone = "recent" if f.turn >= naive else ("mid" if f.turn >= capped else "deep")
        out.append(replace(f, zone=zone))
    return out


def main() -> None:
    """Print the transcript shape, the derived zones, and the gate results."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    turns, facts, spec = build()
    total = sum(tokens.count_turn(t) for t in turns)
    per_turn = total / len(turns)

    console.print(
        f"[bold]transcript[/bold]  {len(turns)} turns, {total:,} tok "
        f"({per_turn:.0f}/turn), {total / config.CONTEXT_BUDGET_TOKENS:.1f}x the "
        f"{config.CONTEXT_BUDGET_TOKENS}-token budget"
    )
    console.print(f"spec hash  {spec_hash()}")

    reach = window_reach(turns)
    t = Table(title="window reach (computed, zero API cost)")
    t.add_column("arm"); t.add_column("oldest turn held", justify="right")
    for name, oldest in reach.items():
        t.add_row(name, str(oldest))
    console.print(t)

    facts = assign_zones(facts, reach)
    counts: dict[str, int] = {}
    for f in facts:
        counts[f.zone] = counts.get(f.zone, 0) + 1
    narrow = max(reach.values())
    wide = min(reach[k] for k in ("cap+pin+window", "+summarize"))
    console.print(
        f"zones  recent >= turn {narrow} (inside every arm's window) | "
        f"mid {wide}-{narrow - 1} (capping reaches it, the naive window does not) | "
        f"deep < turn {wide} (outside every window)"
    )
    console.print(f"probes {len(facts)}: {counts}  headline n = "
                  f"{counts.get('mid', 0) + counts.get('deep', 0)}")

    carriers: dict[str, int] = {}
    for f in facts:
        if f.zone in ("negative", "doc"):
            continue
        carriers[f.carrier] = carriers.get(f.carrier, 0) + 1
    console.print(f"carriers {carriers}")

    failures = run_gates(turns, facts, spec)
    if failures:
        console.print(f"[bold red]{len(failures)} gate failures[/bold red]")
        for f in failures[:20]:
            console.print(f"  {f}")
    else:
        console.print("[bold green]all build gates pass[/bold green]")


if __name__ == "__main__":
    main()
