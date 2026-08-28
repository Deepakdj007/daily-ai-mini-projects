"""Render the scorecard to output/leaderboard.md.

Inputs:  the manifest, the result rows and the validity-gate outcomes
Outputs: a markdown file, tracked in git so a reader can diff two runs

Split out from report.py because the two do different jobs: report.py decides
what the numbers mean, this file decides how they are laid out. A slicing bug
should be findable without reading markdown-generation code.
"""

from __future__ import annotations

from typing import Sequence

from src import config
from src.ladder import BY_NAME
from src.report import CRITERION, _rows_by_arm, by_carrier, decomposition, recall


def write_markdown(manifest: dict, rows: Sequence[dict],
                   checks: Sequence[tuple[str, bool, str]],
                   met: bool, detail: str) -> None:
    """Emit output/leaderboard.md, tracked in git so readers can diff it."""
    by_arm = _rows_by_arm(rows)
    order = [a for a in ("none", "raw", "window", "cap", "pin", "summarize",
                         "full", "no-pin") if a in by_arm]
    out: list[str] = ["# Context layer leaderboard", ""]
    out.append(f"> {CRITERION}")
    out.append("")
    out.append(f"**Result: {'MET' if met else 'NOT MET'}.** {detail}")
    out.append("")
    out.append(
        f"`{manifest['model']}` | scenario `{manifest['scenario']}` | "
        f"budget {manifest['context_budget']} tok | "
        f"summary leakage {manifest.get('facts_in_summary')} | "
        f"cached tokens rate-limit exempt: "
        f"{manifest.get('cached_tokens_rate_limit_exempt')}"
    )
    out += ["", "## Validity gates", "",
            "| check | result | detail |", "| --- | --- | --- |"]
    for name, ok, det in checks:
        out.append(f"| {name} | {'pass' if ok else '**FAIL**'} | {det} |")

    out += ["", "## The ladder", "",
            "| arm | what it adds | out-of-window recall | recent | turns held | assembled tok |",
            "| --- | --- | ---: | ---: | ---: | ---: |"]
    for name in order:
        hit, n = recall(by_arm[name])
        rh, rn = recall(by_arm[name], ("recent",))
        turns = max((r["turns_in_context"] for r in by_arm[name]), default=0)
        tok = max((r["assembled_tokens"] for r in by_arm[name]), default=0)
        label = BY_NAME[name].label if name in BY_NAME else ""
        out.append(
            f"| {name} | {label} | {hit}/{n}"
            + (f" ({hit / n:.0%})" if n else "")
            + f" | {rh}/{rn} | {turns} | {tok:,} |"
        )

    out += ["", "## Did the layer fail, or the model?", "",
            "| arm | present & correct | present & missed | absent & missed | absent & correct |",
            "| --- | ---: | ---: | ---: | ---: |"]
    for name in order:
        d = decomposition(by_arm[name])
        out.append(f"| {name} | {d['present_correct']} | {d['present_missed']} "
                   f"| {d['absent_missed']} | {d['absent_correct']} |")
    out.append("")
    out.append("`absent & missed` is the layer's failure. `present & missed` is the "
               "model's: the fact was in the prompt and it still did not answer.")

    out += ["", "## What the run found that the design did not predict", "",
            "| arm | out-of-window recall, fact in prose | fact in tool output |",
            "| --- | ---: | ---: |"]
    for name in order:
        c = by_carrier(by_arm[name])
        prose, tool = c["prose"], c["tool"]
        out.append(
            f"| {name} | {prose[0]}/{prose[1]} | {tool[0]}/{tool[1]} |"
            if prose[1] or tool[1] else f"| {name} | - | - |"
        )
    out.append("")
    out.append(
        "BM25 matches tokens. A value sitting in a JSON field with a "
        "distinctive name is findable; the same value paraphrased into a "
        "sentence is not. The build gate that forces low lexical overlap "
        "between a probe and the transcript - the gate that stops the "
        "retrieval test being rigged - is precisely what makes prose "
        "invisible to a lexical retriever. Swapping BM25 for embeddings is "
        "the obvious next move, and this table is the reason."
    )
    out.append("")
    config.LEADERBOARD_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")
