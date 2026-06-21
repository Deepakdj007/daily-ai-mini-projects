"""The ReAct tool loop that turns a worker into a real agent.

Each agent gets a bound toolset and DECIDES which tools to call, reads the
results, and loops until it can answer (or hits the iteration cap). This is the
difference between an "agent" and a node that just summarizes a fixed fetch.
Calls go through `ainvoke_with_backoff`, so a free-tier 429 self-heals.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_groq import ChatGroq

from app.agents.llm import _is_rate_limit, ainvoke_with_backoff
from app.config import agent_model
from app.state import Finding


_VERBOSE = False


def set_verbose(verbose: bool) -> None:
    """Toggle printing each agent's tool calls (used by the --verbose CLI flag)."""
    global _VERBOSE
    _VERBOSE = verbose


def _text(content: object) -> str:
    """Flatten a message's content to a clean string."""
    if isinstance(content, list):
        return " ".join(str(p) for p in content).strip()
    return str(content).strip()


async def run_tool_agent(model: ChatGroq, tools: list[BaseTool], system: str,
                         task: str, *, max_iters: int = 4,
                         tool_log: list[str] | None = None) -> str:
    """Run a tool-calling loop and return the agent's final text answer."""
    by_name = {t.name: t for t in tools}
    bound = model.bind_tools(tools)
    messages: list = [SystemMessage(content=system), HumanMessage(content=task)]

    for _ in range(max_iters):
        ai = await ainvoke_with_backoff(bound, messages)
        messages.append(ai)
        if not ai.tool_calls:
            return _text(ai.content)
        for call in ai.tool_calls:
            if tool_log is not None:
                tool_log.append(call["name"])
            tool = by_name.get(call["name"])
            try:
                result = (await tool.ainvoke(call["args"]) if tool
                          else f"unknown tool {call['name']}")
            except Exception as exc:  # noqa: BLE001 — feed the error back to the agent
                result = f"tool error: {exc}"
            messages.append(ToolMessage(content=str(result)[:1500],
                                        tool_call_id=call["id"]))

    # Iteration cap reached — force a tool-free final answer.
    final = await ainvoke_with_backoff(model, messages + [HumanMessage(
        content="Give your final answer now from what you have. Do not call tools.")])
    return _text(final.content)


async def run_agent_finding(name: str, title: str, tools: list[BaseTool],
                            system: str, task: str) -> dict:
    """Run an agent and wrap its answer (or failure) into a partial state dict."""
    log: list[str] = []
    try:
        text = await run_tool_agent(agent_model(), tools, system, task, tool_log=log)
    except Exception as exc:  # noqa: BLE001 — one agent's failure isn't fatal
        return {"findings": [Finding(name, title, f"{title} unavailable: {exc}",
                                     False, _is_rate_limit(exc))],
                "missing": [name]}
    if _VERBOSE:
        print(f"  · {name} called tools: {', '.join(log) or '(none)'}")
    if not text.strip():
        return {"findings": [Finding(name, title, "no output produced", False)],
                "missing": [name]}
    return {"findings": [Finding(name, title, text, True)]}
