"""Item-level and run-level evaluators for the experiment harness.

Inputs: each evaluator receives the raw `output` dict returned by `run_agent`
(answer, token counts, cost_usd, latency_s) plus the item's `expected_output`.
Outputs: `Evaluation` objects that Langfuse attaches as scores on each trace.
"""

import json
import re
import string
from typing import Any

from langfuse import Evaluation

from src.agent import build_groq_client, create_completion_with_retry
from src.config import JUDGE_MODEL

_judge_client = build_groq_client()

JUDGE_PROMPT = """Question: {question}
Expected answer: {expected}
Given answer: {actual}

Judge whether the given answer is correct, allowing for different wording or formatting \
of the same value. Respond with strict JSON only: {{"correct": 0 or 1, "reason": "short reason"}}"""


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace for loose text comparison."""
    lowered = text.lower().strip()
    no_punct = lowered.translate(str.maketrans("", "", string.punctuation))
    return " ".join(no_punct.split())


def exact_match_evaluator(
    *, input: Any, output: Any, expected_output: Any, metadata: Any = None, **kwargs: Any
) -> Evaluation:
    """Normalized substring match between the agent's answer and the expected output."""
    answer = _normalize(str(output.get("answer", "")))
    expected = _normalize(str(expected_output))
    is_match = bool(expected) and expected in answer
    return Evaluation(
        name="exact_match",
        value=1.0 if is_match else 0.0,
        comment=f"expected '{expected}' in answer '{answer}'",
    )


async def llm_judge_evaluator(
    *, input: Any, output: Any, expected_output: Any, metadata: Any = None, **kwargs: Any
) -> Evaluation:
    """Ask a fixed 70b referee model whether the answer is correct, independent of wording."""
    prompt = JUDGE_PROMPT.format(
        question=input, expected=expected_output, actual=output.get("answer", "")
    )
    response = await create_completion_with_retry(
        _judge_client,
        model=JUDGE_MODEL,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
        score = float(parsed.get("correct", 0))
        reason = str(parsed.get("reason", ""))
    except (json.JSONDecodeError, TypeError, ValueError):
        digit_match = re.search(r"[01]", raw)
        score = float(digit_match.group()) if digit_match else 0.0
        reason = f"unparsed judge response: {raw[:100]}"
    return Evaluation(name="llm_judge", value=score, comment=reason)


def cost_evaluator(
    *, input: Any, output: Any, expected_output: Any, metadata: Any = None, **kwargs: Any
) -> Evaluation:
    """Attach the real dollar cost of this task run, computed from token usage."""
    return Evaluation(name="cost_usd", value=output.get("cost_usd", 0.0))


def latency_evaluator(
    *, input: Any, output: Any, expected_output: Any, metadata: Any = None, **kwargs: Any
) -> Evaluation:
    """Attach the wall-clock latency of this task run."""
    return Evaluation(name="latency_s", value=output.get("latency_s", 0.0))


def total_cost_run_evaluator(*, item_results: list[Any], **kwargs: Any) -> Evaluation:
    """Sum the per-item cost evaluations into a run-level total."""
    total = sum(
        evaluation.value
        for result in item_results
        for evaluation in result.evaluations
        if evaluation.name == "cost_usd"
    )
    return Evaluation(name="total_cost_usd", value=total, comment=f"summed across {len(item_results)} items")
