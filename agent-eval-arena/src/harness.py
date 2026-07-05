"""Loads the fixed test set and runs one Langfuse experiment per competing config.

Inputs: data/test_set.json, the AgentConfig list from config.py.
Outputs: one ExperimentResult per config, ready for leaderboard aggregation.
"""

import json
from typing import Any

from langfuse import get_client

from src.agent import build_groq_client, run_agent
from src.config import CONFIGS, TEST_SET_PATH, AgentConfig
from src.evaluators import (
    cost_evaluator,
    exact_match_evaluator,
    latency_evaluator,
    llm_judge_evaluator,
    total_cost_run_evaluator,
)

MAX_CONCURRENCY = 2


def load_test_set() -> list[dict[str, Any]]:
    """Read the fixed test set and reshape it into Langfuse's local-item format."""
    raw_items = json.loads(TEST_SET_PATH.read_text(encoding="utf-8"))
    return [
        {
            "input": item["question"],
            "expected_output": item["expected_output"],
            "metadata": {"id": item["id"], "category": item["category"]},
        }
        for item in raw_items
    ]


def make_task(config: AgentConfig) -> Any:
    """Build the per-config task function the experiment runner calls for every item."""
    client = build_groq_client()

    async def task(*, item: Any, **kwargs: Any) -> dict[str, Any]:
        question = item["input"] if isinstance(item, dict) else item.input
        return await run_agent(client, config, question)

    return task


def run_all_configs() -> list[Any]:
    """Run every config against the fixed test set as its own Langfuse experiment."""
    langfuse = get_client()
    data = load_test_set()
    results = []

    for config in CONFIGS:
        result = langfuse.run_experiment(
            name=config.name,
            description=f"Agent eval arena run for model={config.model}",
            data=data,
            task=make_task(config),
            evaluators=[exact_match_evaluator, llm_judge_evaluator, cost_evaluator, latency_evaluator],
            run_evaluators=[total_cost_run_evaluator],
            max_concurrency=MAX_CONCURRENCY,
        )
        results.append(result)

    langfuse.flush()
    return results
