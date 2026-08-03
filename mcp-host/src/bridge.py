"""
Translates between MCP's tool format and Groq's function-calling format.

Two pure functions, no I/O, so they are easy to reason about and to test.

Inputs: MCP Tool objects and CallToolResult objects.
Outputs: Groq tool schemas and plain-text tool results.
"""

from __future__ import annotations

import json

import mcp.types as types

from src import config

# Groq/OpenAI function names must match ^[a-zA-Z0-9_-]{1,64}$.
MAX_NAME_LEN = 64


def to_groq_tools(tools: dict[str, types.Tool], exposed: list[str]) -> list[dict]:
    """
    Build the Groq tools array from the host's aggregated MCP tools.

    Only names in `exposed` are included. Every schema is resent on every turn, so
    trimming here is what keeps a five-server host inside the free-tier budget.
    """
    allowed = set(exposed)
    schemas: list[dict] = []

    for name, tool in tools.items():
        if name not in allowed or len(name) > MAX_NAME_LEN:
            continue
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": (tool.description or "").strip(),
                    # snake_case in mcp 2.0 — this was inputSchema in v1.
                    "parameters": tool.input_schema,
                },
            }
        )

    return schemas


def _block_to_text(block: types.ContentBlock) -> str:
    """Render one MCP content block as text, naming the ones we can't inline."""
    if isinstance(block, types.TextContent):
        return block.text
    if isinstance(block, types.ImageContent):
        return f"[image: {block.mime_type}]"
    if isinstance(block, types.AudioContent):
        return f"[audio: {block.mime_type}]"
    if isinstance(block, types.ResourceLink):
        return f"[resource: {block.uri}]"
    return f"[{type(block).__name__}]"


def result_to_text(result: types.CallToolResult) -> str:
    """
    Flatten a CallToolResult into the string that goes back to the model.

    Structured output is preferred when a server provides it. Errors are handed
    back as text rather than raised, so the model can read the message and retry
    with corrected arguments.
    """
    if result.structured_content is not None:
        text = json.dumps(result.structured_content, ensure_ascii=False, default=str)
    else:
        text = "\n".join(_block_to_text(b) for b in result.content).strip()

    if not text:
        text = "(the tool returned nothing)"

    if len(text) > config.MAX_RESULT_CHARS:
        text = text[: config.MAX_RESULT_CHARS] + "\n... [truncated]"

    return f"ERROR: {text}" if result.is_error else text
