"""Running the ladder: every case against every arm, resumably.

Inputs:  a profile, a model, and the corpus
Outputs: output/results-<model>.json - one row per (arm, case), plus a manifest

The free tier gives 200,000 tokens a day per model and the full matrix needs
more than that, so the run has to survive being stopped. Rows already in the
results file are skipped, the day's spend is read from the local ledger, and
the harness stops itself at 95% rather than discovering the ceiling as a 429.

Cases are ordered so the headline finishes first. A run interrupted on day one
should still be able to answer the question the project is about.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

from src import cache, config, gates, index, llm, pipeline, score, screen
from src.adversary import build as build_attacks
from src.corpus import Corpus, Question, load
from src.ladder import ALL_ARMS, BY_NAME, PROFILES, Arm


@dataclass(frozen=True, slots=True)
class Case:
    """One question under one condition."""

    qid: str
    cond: str
    dose: int = 0

    @property
    def key(self) -> str:
        return f"{self.qid}|{self.cond}|{self.dose}"


# n per condition, and the day it belongs to. Day 1 is the headline plus the
# clean bound it is judged against; nothing else is needed to state the result.
PLAN: tuple[tuple[str, int, int, int], ...] = (
    # (condition, n questions, dose, day)
    ("poison-p", 38, 3, 1),
    ("clean", 38, 0, 1),
    ("absent", 38, 0, 2),
    ("poison-v", 15, 3, 2),
    ("stale", 15, 0, 2),
    ("inject-overt", 15, 1, 2),
    ("inject-policy", 15, 1, 2),
    ("poison-p", 10, 1, 3),
    ("faq", 8, 0, 3),
    ("saturate", 10, 5, 3),
    ("sametier", 10, 0, 3),
    ("embedded", 10, 0, 3),
)

LITE: tuple[tuple[str, int, int, int], ...] = (
    ("poison-p", 10, 3, 1), ("clean", 10, 0, 1),
    ("absent", 10, 0, 1), ("poison-v", 10, 3, 1),
)
SMOKE: tuple[tuple[str, int, int, int], ...] = (
    ("poison-p", 3, 3, 1), ("clean", 3, 0, 1),
)


def cases(corpus: Corpus, profile: str) -> list[Case]:
    """Build the case list for a profile, headline first."""
    plan = {"smoke": SMOKE, "lite": LITE}.get(profile, PLAN)
    built: list[Case] = []
    seen: set[str] = set()
    for cond, n, dose, _day in sorted(plan, key=lambda row: row[3]):
        pool = corpus.faq_questions if cond == "faq" else corpus.questions
        for question in pool[:n]:
            case = Case(question.qid, cond, dose)
            if case.key not in seen:
                seen.add(case.key)
                built.append(case)
    return built


def _row(arm: Arm, case: Case, question: Question, out: pipeline.Outcome) -> dict:
    """One scorecard line, carrying enough to re-derive every number offline."""
    result = score.outcome(question, case.cond, out.verdict)
    return {
        "arm": arm.name, "qid": case.qid, "cond": case.cond, "dose": case.dose,
        "outcome": result,
        "passed": score.passed(question, case.cond, result),
        "answer": out.verdict.answer, "status": out.verdict.status,
        "reason": out.verdict.reason, "cited": out.verdict.cited,
        "conflicts": list(out.verdict.conflicts),
        "gold": question.gold, "target": question.target,
        "gold_tier": question.gold_tier,
        "gold_retrieved": out.gold_retrieved,
        "attack_reached_reader": out.attack_reached_reader,
        "kept": [p.pid for p in out.kept],
        "dropped": out.dropped,
        "claims": [{"pid": c.pid, "value": c.value, "verified": c.verified,
                    "tier": c.tier, "failed": c.failed, "attempt": c.attempt}
                   for c in out.claims],
        "prompt_tokens": out.prompt_tokens,
        "completion_tokens": out.completion_tokens,
        "attempt": out.attempt,
        "failed": out.failed,
    }


async def run(profile: str = "full", model: str = "", *, resume: bool = True) -> dict:
    """Run the ladder, skipping anything already recorded."""
    model = model or config.CHAT_MODEL
    corpus = load()
    attacks = build_attacks(corpus)
    lookup = {passage.pid: passage for passage in attacks}
    conn = index.connect()

    failures = gates.run_all(conn, corpus, attacks)
    if failures:
        raise SystemExit("build gates failed:\n  " + "\n  ".join(failures))

    payload = _load(model) if resume else {"manifest": {}, "results": []}
    done = {(row["arm"], row["qid"], row["cond"], row["dose"]) for row in payload["results"]}
    rows: list[dict] = list(payload["results"])

    arms = [BY_NAME[name] for name in PROFILES[profile]]
    todo = [(arm, case) for case in cases(corpus, profile) for arm in arms
            if (arm.name, case.qid, case.cond, case.dose) not in done]

    guard = await screen.guard_scores(list(corpus.passages) + attacks)
    print(f"{len(todo)} rows to run ({len(done)} already recorded)")

    stopped = ""
    for number, (arm, case) in enumerate(todo, start=1):
        # spent_today returns (tokens, requests). Both ceilings are real: the
        # request cap bites first on the isolating arms, which make five calls
        # per case instead of one.
        spent, requests = cache.spent_today(model)
        if (spent > config.TOKENS_PER_DAY * config.DAILY_STOP_AT
                or requests > config.REQUESTS_PER_DAY * config.DAILY_STOP_AT):
            stopped = (f"daily budget reached at {spent:,} tokens and {requests:,} "
                       f"requests on {model}; {len(todo) - number + 1} rows left "
                       f"- rerun tomorrow")
            print(stopped)
            break

        question = corpus.question(case.qid)
        out = await pipeline.answer(
            conn, corpus, lookup, question, arm.policy, cond=case.cond,
            dose=case.dose or config.MAX_POISON, arm=arm.name, model=model,
            guard_scores=guard)
        rows.append(_row(arm, case, question, out))
        if number % 10 == 0 or number == len(todo):
            print(f"  {number}/{len(todo)}  {spent:,} tok / {requests:,} req today")
            _write(model, rows, corpus, profile, stopped)

    payload = _write(model, rows, corpus, profile, stopped)
    conn.close()
    return payload


def _write(model: str, rows: list[dict], corpus: Corpus, profile: str, stopped: str) -> dict:
    """Persist results plus the manifest a report needs to trust them."""
    payload = {
        "manifest": {
            "model": model, "profile": profile,
            "arms": [arm.name for arm in ALL_ARMS if arm.name in PROFILES[profile]],
            "corpus": corpus.digest,
            "prompt_version": config.PROMPT_VERSION,
            "temperature": config.TEMPERATURE,
            "reasoning_effort": config.REASONING_EFFORT,
            "top_k": config.TOP_K, "dense_k": config.DENSE_K,
            "guard_block_at": config.GUARD_BLOCK_AT,
            "echo_block_at": config.ECHO_BLOCK_AT,
            "llm_calls": llm.stats(),
            "spent_today": list(cache.spent_today(model)),
            "stopped": stopped,
        },
        "results": rows,
    }
    path = config.results_path(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _load(model: str) -> dict:
    """Previous rows, but only if they were produced by this same experiment."""
    path = config.results_path(model)
    if not path.exists():
        return {"manifest": {}, "results": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest = payload.get("manifest", {})
    corpus = load()
    stale = (manifest.get("corpus") != corpus.digest
             or manifest.get("prompt_version") != config.PROMPT_VERSION)
    if stale:
        print("previous results predate the current corpus or prompts - starting over")
        return {"manifest": {}, "results": []}
    return payload


if __name__ == "__main__":
    asyncio.run(run("smoke"))
