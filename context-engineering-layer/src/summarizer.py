"""The frozen running summary, and the gate that stops it leaking answers.

Inputs:  the dropped-middle text and a token budget
Outputs: data/summary_fixture.json - the summary, its provenance and its hash

The summary is generated ONCE and frozen. Regenerating it per probe would be
expensive and non-deterministic, and hashing a fresh one into the replay key
every run would silently miss the cache and burn the whole daily budget.

The gate is the important part. A summariser compressing an incident transcript
preferentially copies incident ids, error codes and thresholds - which is
exactly what a support summary is FOR, and exactly what has been planted. Left
unchecked it hands every summarising arm the answer key, produces a perfect
monotone ladder, and makes the deep zone look easy for entirely the wrong
reason. So the fixture is grepped for every gold value before it is trusted.
"""

from __future__ import annotations

import asyncio
import hashlib
import json

from src import config, llm

SUMMARY_PROMPT = (
    "You are keeping a running summary of a long incident-response session. "
    "Fold the new material below into the summary so far. Describe the SHAPE "
    "of the conversation for an engineer picking it up mid-shift: who is "
    "involved, what has been covered, what is still open. Do NOT restate "
    "specific identifiers, amounts, codes, timestamps or reference numbers - "
    "those stay in the transcript. Reply with the updated summary only, at "
    "most {budget} tokens of prose."
)

# The dropped middle is ~14,000 tokens, which is nearly twice the whole
# per-minute bucket, so it cannot be summarised in one call. That is not a
# limitation to work around - it is what a running summary actually is. The
# summary is folded forward chunk by chunk, the way a real agent would build it
# as the conversation grew.
CHUNK_TOKENS = 2_500

NL = "\n"


def _hash(text: str) -> str:
    """Content hash, stamped into the cache key so an edit invalidates it."""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _chunks(text: str, chunk_tokens: int) -> list[str]:
    """Split the dropped middle into pieces that fit the per-minute bucket."""
    from src import tokens as tok

    lines = text.splitlines()
    out: list[str] = []
    current: list[str] = []
    spent = 0
    for line in lines:
        cost = tok.count_text(line)
        if current and spent + cost > chunk_tokens:
            out.append(NL.join(current))
            current, spent = [], 0
        current.append(line)
        spent += cost
    if current:
        out.append(NL.join(current))
    return out


async def generate(dropped_text: str, budget: int) -> dict[str, object]:
    """Fold the dropped middle into one running summary, chunk by chunk."""
    from src import tokens as tok

    system = SUMMARY_PROMPT.format(budget=budget)
    running = ""
    calls = 0
    for i, chunk in enumerate(_chunks(dropped_text, CHUNK_TOKENS)):
        user = (
            (f"Summary so far:{NL}{running}{NL}{NL}" if running else "")
            + f"New material:{NL}{chunk}"
        )
        # gpt-oss spends completion budget on reasoning before it writes a
        # word, so the ceiling has to cover both or the content comes back
        # empty - which reads as a refusal rather than as starvation.
        result = await llm.complete(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            cache_key=f"summary-fold-{_hash(chunk)}-{budget}-{i}",
            local_prompt_tokens=tok.count_text(system + user, config.TOKENIZER_FUDGE),
            max_completion_tokens=1_024,
        )
        calls += 1
        if result.text.strip():
            running = result.text.strip()
        else:
            raise RuntimeError(
                f"summariser returned nothing on chunk {i} "
                f"({result.finish_reason}: {result.error or 'empty content'}). "
                f"Refusing to freeze an empty fixture - an empty summary would "
                f"make the summarising arms look identical to the rung below."
            )
    return {
        "text": running,
        "hash": _hash(running),
        "model": config.CHAT_MODEL,
        "temperature": config.TEMPERATURE,
        "budget_tokens": budget,
        "fold_calls": calls,
        "prompt": system,
    }


def leaked(summary: str, gold_values: list[str]) -> list[str]:
    """Which planted answers the summary restated verbatim."""
    from src.grader import matches

    return [v for v in gold_values if matches(v, summary)]


def load(gold_values: list[str] | None = None) -> dict[str, object] | None:
    """The frozen fixture, if it has been generated.

    Recomputes the leakage figure when the gold values are supplied, so the
    manifest always records how much of the answer key the summary restated -
    a run whose summarising arms were reading off an oracle has to be
    recognisable from the results file alone, long after the fact.
    """
    if not config.SUMMARY_FIXTURE_PATH.exists():
        return None
    fixture = json.loads(config.SUMMARY_FIXTURE_PATH.read_text(encoding="utf-8"))
    if gold_values:
        hits = leaked(str(fixture["text"]), gold_values)
        fixture["facts_in_summary"] = f"{len(hits)}/{len(gold_values)}"
        fixture["leaked_values"] = hits
    return fixture


def save(fixture: dict[str, object]) -> None:
    """Freeze the fixture to disk."""
    config.SUMMARY_FIXTURE_PATH.write_text(
        json.dumps(fixture, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def as_fn(fixture: dict[str, object]):
    """Adapt the frozen fixture to the SummarizeFn the layer expects."""
    text = str(fixture["text"])

    def _summarize(_dropped: str, _budget: int) -> str:
        return text

    return _summarize


async def build(force: bool = False) -> dict[str, object]:
    """Generate the fixture if needed, then gate it on leakage."""
    from src.scenario import build as build_scenario

    turns, facts, _ = build_scenario()
    gold = [f.value for f in facts if f.zone not in ("negative", "doc")]

    fixture = None if force else load()
    if fixture is None:
        dropped = "\n".join(f"[turn {t.index}] {t.text()}" for t in turns[:-10])
        fixture = await generate(dropped, config.SUMMARY_BUDGET_TOKENS)
        save(fixture)

    hits = leaked(str(fixture["text"]), gold)
    fixture["facts_in_summary"] = f"{len(hits)}/{len(gold)}"
    fixture["leaked_values"] = hits
    return fixture


if __name__ == "__main__":
    from rich.console import Console

    console = Console()
    fx = asyncio.run(build())
    console.print(f"[bold]summary fixture[/bold]  {fx['hash']}  "
                  f"model={fx['model']}  budget={fx['budget_tokens']}")
    console.print(f"\n{fx['text']}\n")
    hits = fx["leaked_values"]
    console.print(f"facts_in_summary: [bold]{fx['facts_in_summary']}[/bold]")
    if hits:
        console.print(f"[yellow]leaked:[/yellow] {hits}")
    LIMIT = 3
    if len(hits) > LIMIT:
        console.print(
            f"[bold red]GATE FAILED[/bold red] - more than {LIMIT} planted "
            f"answers are restated verbatim. Every summarising arm would be "
            f"reading them off an oracle. Tighten the prompt or shrink the "
            f"budget and regenerate with --force."
        )
    else:
        console.print("[bold green]leakage gate passes[/bold green]")
