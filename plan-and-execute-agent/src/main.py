"""CLI entry point: run the plan-execute agent, the ReAct baseline, or both.

    PYTHONPATH=. uv run python -m src.main "your goal here"
    PYTHONPATH=. uv run python -m src.main --agent plan "your goal here"
    PYTHONPATH=. uv run python -m src.main --agent react "your goal here"

With --agent both (the default) it runs each agent on the same goal and prints a
side-by-side table of LLM calls, tool calls, steps, and wall time.
"""

import argparse
import asyncio
import sys
import time

from src.config import require_keys
from src.graph import build_graph
from src.react_agent import run_react
from src.react_loop import Metrics

# Windows consoles default to cp1252, which cannot encode the arrows, em-dashes,
# and non-breaking hyphens the model returns. Force UTF-8 so printing never crashes.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# A multi-hop goal that rewards planning: two lookups feeding one calculation.
DEFAULT_TASK = (
    "What is the height difference in meters between the tallest mountain in Japan "
    "and the tallest mountain in South Korea?"
)


async def run_plan_execute(task: str) -> tuple[str, Metrics, int]:
    """Stream the plan-execute graph, printing the plan and each step live.

    Returns (final answer, metrics, number of executed steps).
    """
    metrics = Metrics()
    graph = build_graph(metrics)
    steps_done = 0
    final_answer = ""

    async for update in graph.astream({"task": task}, {"recursion_limit": 50}):
        for node, delta in update.items():
            if node == "planner":
                print("\nPLAN:")
                for i, step in enumerate(delta["plan"], 1):
                    print(f"  {i}. {step}")
            elif node == "executor":
                step, result = delta["past_steps"][-1]
                steps_done += 1
                print(f"\n[step {steps_done}] {step}")
                print(f"  = {result}")
            elif node == "replanner":
                if delta.get("response"):
                    final_answer = delta["response"]
                elif delta.get("plan"):
                    print(f"  (replanner: {len(delta['plan'])} step(s) remaining)")

    return final_answer, metrics, steps_done


def _print_report(name: str, answer: str, metrics: Metrics, elapsed: float,
                  steps: int | None) -> None:
    """Print one agent's result block."""
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    print(f"ANSWER: {answer}")
    step_line = f" | steps: {steps}" if steps is not None else ""
    print(
        f"llm calls: {metrics.llm_calls} | tool calls: {metrics.tool_calls}"
        f"{step_line} | time: {elapsed:.1f}s"
    )
    print(f"tools used: {', '.join(metrics.tools_used) or '(none)'}")


def _print_failure(name: str, exc: Exception) -> None:
    """Report that one agent failed without taking the whole run down with it."""
    msg = str(exc)
    if "rate_limit" in msg or "429" in msg or "tokens per day" in msg.lower():
        note = (
            "Groq free-tier limit reached (per-day tokens or per-minute rate). "
            "Wait for the reset shown in the message, or use a key from a different "
            "Groq account — a different key in the SAME account shares the same limit."
        )
    else:
        note = msg
    print(f"\n{'=' * 60}\n{name} — DID NOT FINISH\n{'=' * 60}\n{note}")


def _print_comparison(plan_row: tuple, react_row: tuple) -> None:
    """Print the side-by-side metrics table (plain text, no rich dependency)."""
    header = f"{'metric':<14}{'plan-execute':>16}{'react':>16}"
    print(f"\n{'#' * 46}\nCOMPARISON\n{'#' * 46}")
    print(header)
    print("-" * 46)
    labels = ["llm calls", "tool calls", "steps", "time (s)"]
    for label, p, r in zip(labels, plan_row, react_row):
        print(f"{label:<14}{str(p):>16}{str(r):>16}")


async def _main(task: str, which: str) -> None:
    """Run the requested agent(s) on the task."""
    require_keys()
    print(f"GOAL: {task}")

    plan_res = react_res = None

    if which in ("plan", "both"):
        print(f"\n{'>' * 20} PLAN-AND-EXECUTE {'>' * 20}")
        start = time.perf_counter()
        try:
            answer, metrics, steps = await run_plan_execute(task)
            elapsed = time.perf_counter() - start
            _print_report("PLAN-AND-EXECUTE", answer, metrics, elapsed, steps)
            plan_res = (metrics.llm_calls, metrics.tool_calls, steps, f"{elapsed:.1f}")
        except Exception as exc:  # noqa: BLE001 — one agent's failure isn't fatal
            _print_failure("PLAN-AND-EXECUTE", exc)

    if which in ("react", "both"):
        print(f"\n{'>' * 20} REACT BASELINE {'>' * 20}")
        start = time.perf_counter()
        try:
            answer, metrics = await run_react(task)
            elapsed = time.perf_counter() - start
            _print_report("REACT", answer, metrics, elapsed, None)
            react_res = (metrics.llm_calls, metrics.tool_calls, "-", f"{elapsed:.1f}")
        except Exception as exc:  # noqa: BLE001
            _print_failure("REACT", exc)

    if plan_res and react_res:
        _print_comparison(plan_res, react_res)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan-and-execute agent vs a plain ReAct agent."
    )
    parser.add_argument("task", nargs="?", default=DEFAULT_TASK,
                        help="The goal to solve (defaults to a multi-hop demo).")
    parser.add_argument("--agent", choices=["plan", "react", "both"], default="both",
                        help="Which agent(s) to run. Default: both.")
    args = parser.parse_args()
    asyncio.run(_main(args.task, args.agent))


if __name__ == "__main__":
    main()
