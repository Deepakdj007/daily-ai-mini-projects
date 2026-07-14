"""The shared tool-calling loop, plus a Metrics counter used for the comparison.

Both agents run through run_tool_agent. The plan-execute agent calls it once per
step (with a small cap); the ReAct baseline calls it once for the whole goal (with
a big cap). Same primitive, so the only difference on trial is planning vs not.
"""

import asyncio
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_groq import ChatGroq


@dataclass
class Metrics:
    """Running tallies for one agent run — the numbers the comparison table shows."""

    llm_calls: int = 0
    tool_calls: int = 0
    tools_used: list[str] = field(default_factory=list)


async def _ainvoke_with_backoff(model, messages: list, metrics: Metrics) -> AIMessage:
    """Invoke the model, retrying on Groq free-tier 429s with growing backoff."""
    delay = 2.0
    for attempt in range(4):
        try:
            ai = await model.ainvoke(messages)
            metrics.llm_calls += 1
            return ai
        except Exception as exc:  # noqa: BLE001
            is_rate_limit = "429" in str(exc) or "rate limit" in str(exc).lower()
            if not is_rate_limit or attempt == 3:
                raise
            await asyncio.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def _text(content: object) -> str:
    """Flatten a message's content (sometimes a list of parts) to a clean string."""
    if isinstance(content, list):
        return " ".join(str(part) for part in content).strip()
    return str(content).strip()


async def run_tool_agent(
    model: ChatGroq,
    tools: list[BaseTool],
    system: str,
    task: str,
    *,
    max_iters: int,
    metrics: Metrics,
    verbose: bool = False,
) -> str:
    """Run a bind_tools ReAct loop and return the agent's final text answer.

    Loops up to max_iters: ask the model, run any tool calls, feed results back.
    When the model stops calling tools it has its answer. If it hits the cap it is
    forced to answer from what it has. Every LLM and tool call is counted in metrics.
    """
    by_name = {t.name: t for t in tools}
    bound = model.bind_tools(tools)
    messages: list = [SystemMessage(content=system), HumanMessage(content=task)]

    for _ in range(max_iters):
        ai = await _ainvoke_with_backoff(bound, messages, metrics)
        messages.append(ai)
        if not ai.tool_calls:
            return _text(ai.content)
        for call in ai.tool_calls:
            metrics.tool_calls += 1
            metrics.tools_used.append(call["name"])
            if verbose:
                print(f"    -> {call['name']}({call['args']})")
            tool = by_name.get(call["name"])
            try:
                result = (
                    await tool.ainvoke(call["args"]) if tool
                    else f"error: unknown tool {call['name']}"
                )
            except Exception as exc:  # noqa: BLE001 — feed the error back to the agent
                result = f"error: {exc}"
            if verbose:
                print(f"    <- {str(result)[:120]}")
            messages.append(ToolMessage(content=str(result)[:1500],
                                        tool_call_id=call["id"]))

    # Cap reached: force a final answer from what the agent gathered. We keep the
    # tools BOUND here on purpose. gpt-oss is a reasoning model that will emit a
    # tool call in its raw output even when asked not to; if we sent the unbound
    # model (tool_choice="none"), Groq rejects that with a 400 tool_use_failed. With
    # tools still bound the call always succeeds — we simply ignore any tool calls
    # and read the text. If it produced no text, fall back to the last useful message.
    final = await _ainvoke_with_backoff(
        bound,
        messages + [HumanMessage(content="Give your final answer now in plain text.")],
        metrics,
    )
    text = _text(final.content)
    if text:
        return text
    for msg in reversed(messages):
        if isinstance(msg, (AIMessage, ToolMessage)) and _text(msg.content):
            return _text(msg.content)
    return "no answer produced"
