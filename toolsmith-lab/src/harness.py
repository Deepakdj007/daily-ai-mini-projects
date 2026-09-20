"""Running the ladder: every task against every arm, resumably.

Inputs:  a profile, and whatever is already in the results file
Outputs: output/results-<model>.json, written after every task

The free tier gives 200,000 tokens a day and the full matrix needs several
times that, so the run has to survive being stopped and picked up tomorrow.
Rows already in the file are skipped, the day's spend comes from the local
ledger, and the harness stops itself at 95% rather than discovering the ceiling
as a wall of 429s.

The arm order is not the ladder order. It is the order that answers the
question soonest: the headline pair and both floors go first, so a run that
never gets past day two still has something to report. A ladder run in rung
order spends its first day on arms that only make the picture prettier.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from src import agent, cache, config, corpus, ladder, llm, registry

# Which families each profile uses. smoke is one family so the whole loop can
# be watched in one sitting; lite is three, which is what one day buys.
PROFILE_FAMILIES: dict[str, tuple[str, ...] | None] = {
    "smoke": ("gstin",),
    "lite": ("gstin", "isoweek", "words"),
    "full": None,
}

# The order arms are run in, and the day each is expected to land on.
PLAN: tuple[tuple[str, int], ...] = (
    ("persist", 1), ("none", 1), ("fixed", 1),
    ("holdout", 2),
    ("tested", 3), ("inline", 4), ("write", 5),
)


def _ordered(profile: str) -> list[ladder.Arm]:
    """The profile's arms, in the order that answers the question soonest."""
    wanted = {arm.name for arm in ladder.arms_for(profile)}
    ranked = [name for name, _ in PLAN if name in wanted]
    return [ladder.BY_NAME[name] for name in ranked]


def _load(model: str) -> tuple[list[dict], dict]:
    """Rows already run, unless the corpus or the oracle has moved since."""
    path = config.results_path(model)
    if not path.exists():
        return [], {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest = payload.get("manifest", {})
    stamp = corpus.stamp()
    if (manifest.get("corpus") != stamp["corpus"]
            or manifest.get("oracle") != stamp["oracle"]
            or manifest.get("prompt_version") != config.PROMPT_VERSION):
        print(f"  {path.name} predates the current corpus or prompts - starting over")
        return [], {}
    return payload.get("results", []), manifest


def _write(model: str, rows: list[dict], profile: str, stopped: str = "") -> dict:
    """Results plus a manifest saying exactly what produced them."""
    payload = {
        "manifest": {
            "model": model or config.CHAT_MODEL,
            "profile": profile,
            **corpus.stamp(),
            "prompt_version": config.PROMPT_VERSION,
            "temperature": config.TEMPERATURE,
            "reasoning_effort": config.REASONING_EFFORT,
            "max_rounds": config.MAX_ROUNDS,
            "max_repairs": config.MAX_REPAIR_ATTEMPTS,
            "budgets": config.MAX_COMPLETION_TOKENS,
            "llm_calls": llm.stats(),
            "ledger": cache.ledger_rows(),
            "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "stopped": stopped,
        },
        "results": rows,
    }
    config.ensure_dirs()
    config.results_path(model).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def _budget_left(model: str) -> int:
    """Tokens still safe to spend today, by the local ledger."""
    spent, _ = cache.spent_today(model or config.CHAT_MODEL)
    return int(config.TOKENS_PER_DAY * config.DAILY_STOP_AT) - spent


async def run(profile: str = "lite", model: str = "", *, resume: bool = True,
              only_arm: str = "") -> dict:
    """Run the ladder, skipping what is already done, stopping at the budget."""
    rows, _ = _load(model) if resume else ([], {})
    done = {(row["arm"], row["task_id"]) for row in rows}
    arms = [a for a in _ordered(profile) if not only_arm or a.name == only_arm]
    tasks = corpus.tasks_for(PROFILE_FAMILIES.get(profile))
    families = {entry["family"]: entry for entry in corpus.families()}

    total = len(arms) * len(tasks)
    print(f"{profile}: {len(arms)} arms x {len(tasks)} tasks = {total} runs, "
          f"{len(done)} already done, {_budget_left(model):,} tokens left today")

    stopped = ""
    for arm in arms:
        reg = registry.Registry(persist=arm.policy.persist, arm=arm.name)
        if arm.policy.persist:
            # A fresh sweep starts from an empty toolbox; a resumed one has to
            # pick up the tools the interrupted half already wrote, or the
            # tasks after the break would each write their own and the reuse
            # figure would measure where the run was interrupted.
            if any(row["arm"] == arm.name for row in rows):
                found = reg.load()
                print(f"  {arm.name}: resuming with {found} tool(s) from generated/")
            else:
                registry.clear_generated()
        for task in tasks:
            if (arm.name, task["id"]) in done:
                continue
            if _budget_left(model) <= 0:
                stopped = "daily budget"
                break
            row = await agent.run_task(task, families[task["family"]],
                                       arm=arm.name, policy=arm.policy, reg=reg)
            rows.append(row)
            _write(model, rows, profile, stopped)
            mark = "ok " if row["correct"] else "   "
            print(f"  {mark}{arm.name:<8} {task['id']:<11} {row['outcome']:<12} "
                  f"{row['completion_tokens']:>5} completion, "
                  f"{row['writes']}w/{row['registered']}r"
                  + (f", reused {row['reused_from']}" if row["reused"] else ""))
            if row.get("stopped") == "tpd_exhausted":
                stopped = "tokens per day exhausted"
                break
        if stopped:
            break

    payload = _write(model, rows, profile, stopped)
    if stopped:
        print(f"\nstopped: {stopped}. Run the same command tomorrow to continue.")
    else:
        print(f"\ndone: {len(rows)} rows in {config.results_path(model).name}")
    return payload


if __name__ == "__main__":
    for profile in PROFILE_FAMILIES:
        arms = _ordered(profile)
        tasks = corpus.tasks_for(PROFILE_FAMILIES[profile])
        print(f"{profile:<6} {len(arms)} arms x {len(tasks):>2} tasks = "
              f"{len(arms) * len(tasks):>3} runs   "
              f"[{', '.join(a.name for a in arms)}]")
    assert _ordered("full")[0].name == "persist", "the headline arm must run first"
    assert {a.name for a in _ordered("full")} == set(ladder.PROFILES["full"])
    print(f"\ntoday's remaining budget: {_budget_left(''):,} tokens")
    print("OK - harness")
