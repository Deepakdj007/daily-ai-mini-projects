"""The plain ReAct baseline: one tool loop over the whole goal, no planning.

Same model, same tools, same loop primitive as the plan-execute agent's executor —
just no planner and no replanner. This is the control the comparison runs against.
"""

from src.config import MAX_STEPS, MAX_TOOL_ITERS, make_llm
from src.prompts import REACT_SYSTEM
from src.react_loop import Metrics, run_tool_agent
from src.tools import ALL_TOOLS


async def run_react(task: str, *, verbose: bool = True) -> tuple[str, Metrics]:
    """Run the ReAct agent on the goal and return (answer, metrics).

    The iteration cap matches the plan-execute agent's total budget (steps x per-step
    tool iterations), so neither agent gets an unfair number of chances.
    """
    metrics = Metrics()
    answer = await run_tool_agent(
        make_llm(), ALL_TOOLS, REACT_SYSTEM, task,
        max_iters=MAX_STEPS * MAX_TOOL_ITERS, metrics=metrics, verbose=verbose,
    )
    return answer, metrics
