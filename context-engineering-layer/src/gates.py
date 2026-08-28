"""Every build-time check, in one place, costing nothing to run.

Inputs:  the generated transcript, the probes and the raw spec
Outputs: a list of failures - empty when the scenario is trustworthy

A probe that can be answered some other way measures nothing, and it fails
silently: the arm scores well, the ladder looks monotone, and the scorecard is
a lie that nobody has a reason to check. These gates run before any token is
spent, which is the only time catching it is cheap.
"""

from __future__ import annotations

from typing import Any, Sequence

from src import config, tokens
from src.types import Fact


def gates(turns: tuple[Turn, ...], facts: list[Fact], spec: dict[str, Any]) -> list[str]:
    """Every build-time check. Returns the list of failures."""
    from src.grader import gold_pattern, matches, normalize

    failures: list[str] = []
    haystack = spec["system_prompt"] + "\n" + "\n".join(spec["pinned_facts"]) + "\n"
    haystack += "\n".join(t.text() for t in turns)
    norm_hay = normalize(haystack)

    planted = [f for f in facts if f.zone not in ("negative", "doc")]
    for f in planted:
        # Counted with the SAME boundary-anchored matcher that grades answers
        # and decides fact_present. A plain substring count would flag "4712"
        # inside "184712" as a second occurrence, and a gate that disagrees
        # with the grader reports failures the scorecard will never see.
        hits = len(gold_pattern(f.value).findall(norm_hay))
        if hits != 1:
            failures.append(f"gate1 {f.id}: value {f.value!r} occurs {hits} times, want 1")

    spec_by_id = {f["id"]: f for f in spec["facts"]}
    for f in facts:
        entry = spec_by_id.get(f.id)
        if entry and entry.get("anchor") and matches(f.value, entry["anchor"]):
            failures.append(
                f"gate6 {f.id}: the anchor states the value, so the fact "
                f"survives even when the tool output is elided"
            )
        if entry and (entry.get("statement") or entry.get("anchor")):
            probe_words = set(normalize(entry["probe"]).split())
            frame_words = set(normalize(
                entry.get("statement") or entry["anchor"]).split())
            common = probe_words & frame_words
            heavy = {w for w in common if len(w) > 4}
            if len(heavy) > 2:
                failures.append(
                    f"gate7 {f.id}: probe and framing share {sorted(heavy)} - "
                    f"BM25 is being handed the answer"
                )

    for f in facts:
        if f.zone != "doc":
            continue
        body = spec["system_prompt"] if f.schema == "system" else "\n".join(spec["pinned_facts"])
        if not matches(f.value, body):
            failures.append(f"gate2 {f.id}: not present in its declared {f.schema} source")
        transcript = "\n".join(t.text() for t in turns)
        if matches(f.value, transcript):
            failures.append(f"gate2 {f.id}: doc answer also appears in the transcript")

    total = sum(tokens.count_turn(t) for t in turns)
    if total < 4 * config.CONTEXT_BUDGET_TOKENS:
        failures.append(
            f"gate3: transcript {total} tok is under 4x the {config.CONTEXT_BUDGET_TOKENS} budget"
        )

    for f in facts:
        if f.zone in ("negative", "doc"):
            continue
        if f.answer_space < 1000:
            failures.append(f"gate5 {f.id}: answer space {f.answer_space} under 1000")

    for n in spec["negatives"]:
        pass  # nothing to plant, by definition

    return failures
