"""The three plan-execute nodes and the loop router.

planner  -> decompose the goal into steps
executor -> run the first step with tools, record its result
replanner-> given results so far, either finish or hand back the remaining steps

Each node takes a Metrics object (bound in graph.py) so planner/replanner LLM calls
are counted alongside the executor's tool-loop calls.
"""

from langgraph.graph import END

from src.config import MAX_STEPS, MAX_TOOL_ITERS, make_llm
from src.prompts import EXECUTOR_SYSTEM, PLANNER_SYSTEM, REPLANNER_SYSTEM
from src.react_loop import Metrics, run_tool_agent
from src.state import Act, Plan, PlanExecuteState
from src.tools import ALL_TOOLS


def _structured(schema):
    """A Groq LLM that returns `schema` via json_schema mode.

    gpt-oss is a reasoning model: the default tool-calling structured-output path
    makes it emit prose and Groq rejects it with 400 tool_use_failed. json_schema
    mode constrains the raw response instead, which it follows reliably.
    """
    return make_llm().with_structured_output(schema, method="json_schema")


async def planner_node(state: PlanExecuteState, *, metrics: Metrics) -> dict:
    """Turn the goal into an ordered step list."""
    plan: Plan = await _structured(Plan).ainvoke(
        f"{PLANNER_SYSTEM}\n\nGoal: {state['task']}"
    )
    metrics.llm_calls += 1
    return {"plan": plan.steps}


def _format_past(past_steps: list[tuple[str, str]]) -> str:
    """Render completed (step, result) pairs for the replanner prompt."""
    if not past_steps:
        return "(none yet)"
    return "\n".join(f"- {step} -> {result}" for step, result in past_steps)


async def executor_node(state: PlanExecuteState, *, metrics: Metrics) -> dict:
    """Execute the first step of the current plan with the shared tool loop."""
    plan = state["plan"]
    current = plan[0]
    plan_text = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(plan))
    task = (
        f"Overall goal: {state['task']}\n\n"
        f"Full plan:\n{plan_text}\n\n"
        f"Do ONLY this step now: {current}"
    )
    result = await run_tool_agent(
        make_llm(), ALL_TOOLS, EXECUTOR_SYSTEM, task,
        max_iters=MAX_TOOL_ITERS, metrics=metrics, verbose=True,
    )
    return {"past_steps": [(current, result)]}


async def replanner_node(state: PlanExecuteState, *, metrics: Metrics) -> dict:
    """Decide whether the goal is answered, or return the steps still to do."""
    prompt = (
        f"{REPLANNER_SYSTEM}\n\n"
        f"Goal: {state['task']}\n\n"
        f"Completed steps and their results:\n{_format_past(state.get('past_steps', []))}"
    )
    act: Act = await _structured(Act).ainvoke(prompt)
    metrics.llm_calls += 1
    if act.done:
        return {"response": act.answer, "plan": []}
    return {"plan": act.remaining_steps}


def should_end(state: PlanExecuteState) -> str:
    """Loop back to the executor, or stop.

    Stop when the replanner produced a final answer, or when it ran out of steps,
    or when we've executed MAX_STEPS (a hard backstop against an endless replan loop).
    """
    if state.get("response"):
        return END
    if not state.get("plan"):
        return END
    if len(state.get("past_steps", [])) >= MAX_STEPS:
        return END
    return "executor"
