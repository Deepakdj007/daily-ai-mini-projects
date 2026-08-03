"""
The tool-calling loop that drives the five servers from one prompt.

Ask Groq, run whatever tools it asks for, feed the results back, repeat until it
answers in prose. Yields events instead of printing, so the UI decides how to
render and this file stays free of Streamlit.

Inputs: an MCPHost and the running message list.
Outputs: a stream of (kind, payload) events; the message list is updated in place.
"""

from __future__ import annotations

import json
from typing import Any, Iterator

from src import config
from src.bridge import result_to_text, to_groq_tools
from src.host import MCPHost
from src.inventory import NAME_SEP

# Event kinds yielded to the UI.
TEXT = "text"
TOOL_CALL = "tool_call"
TOOL_RESULT = "tool_result"
ERROR = "error"

Event = tuple[str, Any]


def _arguments(raw: str) -> dict:
    """Parse the JSON string Groq puts in function.arguments, tolerating an empty one."""
    if not raw or not raw.strip():
        return {}
    return json.loads(raw)


def _run_one_tool(host: MCPHost, call: Any) -> Iterator[Event]:
    """Execute a single tool call and yield its result event plus the tool message."""
    name = call.function.name
    server, _, tool = name.partition(NAME_SEP)

    try:
        arguments = _arguments(call.function.arguments)
    except json.JSONDecodeError as exc:
        text = f"ERROR: arguments were not valid JSON ({exc})"
        yield TOOL_RESULT, {"id": call.id, "server": server, "tool": tool, "text": text}
        return

    yield TOOL_CALL, {"id": call.id, "server": server, "tool": tool, "arguments": arguments}

    try:
        text = result_to_text(host.call_tool(name, arguments))
    except Exception as exc:
        # Hand the failure back as text so the model can correct itself.
        text = f"ERROR: {type(exc).__name__}: {exc}"

    yield TOOL_RESULT, {"id": call.id, "server": server, "tool": tool, "text": text}


def _ask(host: MCPHost, client: Any, messages: list[dict], tools: list[dict] | None) -> Any:
    """One Groq call, run on the host loop. Omitting tools forces a prose answer."""
    kwargs: dict[str, Any] = {
        "model": config.MODEL,
        "messages": messages,
        # tool_choice is left at auto: gpt-oss does not reliably honour "required",
        # and reasoning_format is rejected outright when tools are in play.
        "temperature": 0.2,
        "max_completion_tokens": 4096,
    }
    if tools:
        kwargs["tools"] = tools
    return host.run(client.chat.completions.create(**kwargs), timeout=config.TOOL_TIMEOUT)


def run_turn(host: MCPHost, client: Any, messages: list[dict]) -> Iterator[Event]:
    """
    Drive one user turn to completion, appending every message it produces.

    Loops until the model answers without asking for tools, or MAX_STEPS is hit.
    """
    tools = to_groq_tools(host.tools, host.exposed_names)

    for _ in range(config.MAX_STEPS):
        try:
            response = _ask(host, client, messages, tools)
        except Exception as exc:
            yield ERROR, f"{type(exc).__name__}: {exc}"
            return

        message = response.choices[0].message
        calls = message.tool_calls or []

        # The assistant message must be appended verbatim, tool_calls included,
        # or the tool replies below have nothing to attach to.
        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                **(
                    {
                        "tool_calls": [
                            {
                                "id": c.id,
                                "type": "function",
                                "function": {
                                    "name": c.function.name,
                                    "arguments": c.function.arguments,
                                },
                            }
                            for c in calls
                        ]
                    }
                    if calls
                    else {}
                ),
            }
        )

        if message.content:
            yield TEXT, message.content

        if not calls:
            return

        for call in calls:
            for kind, payload in _run_one_tool(host, call):
                if kind == TOOL_RESULT:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": payload["id"],
                            "content": payload["text"],
                        }
                    )
                yield kind, payload

    # Out of rounds. Ask once more with no tools attached, so the user gets a real
    # answer built from what we did gather instead of a bare error.
    yield ERROR, f"hit the {config.MAX_STEPS}-round tool limit — summarising what came back"
    try:
        final = _ask(host, client, messages, tools=None).choices[0].message.content or ""
    except Exception as exc:
        yield ERROR, f"{type(exc).__name__}: {exc}"
        return

    messages.append({"role": "assistant", "content": final})
    if final:
        yield TEXT, final
