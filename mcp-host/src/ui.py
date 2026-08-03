"""
Streamlit rendering for the MCP host.

Kept apart from app.py so the page wiring stays readable and the agent loop
never has to know what a widget is.

Inputs: an MCPHost and transcript entries.
Outputs: rendered sidebar, tool-call blocks, and conversation history.
"""

from __future__ import annotations

import streamlit as st

from src import config
from src.bridge import to_groq_tools
from src.host import MCPHost
from src.inventory import NAME_SEP

MAX_RENDER_CHARS = 600


def render_call(entry: dict) -> None:
    """Render one tool call and its result inside a bordered block."""
    with st.container(border=True):
        st.markdown(f"**{entry['server']}** → `{entry['tool']}`")
        if entry.get("arguments"):
            st.json(entry["arguments"], expanded=False)

        result = entry.get("result", "")
        if result.startswith("ERROR:"):
            st.error(result[:MAX_RENDER_CHARS])
        elif result:
            clipped = result[:MAX_RENDER_CHARS]
            st.caption(clipped + ("…" if len(result) > MAX_RENDER_CHARS else ""))


def render_transcript(transcript: list[dict]) -> None:
    """Replay the conversation so far — Streamlit reruns the script on every action."""
    for item in transcript:
        with st.chat_message(item["role"]):
            for call in item.get("calls", []):
                render_call(call)
            if item["content"]:
                st.markdown(item["content"])


def _render_server_row(status) -> None:
    """Draw one server's status line and its tool list."""
    if not status.connected:
        st.markdown(f"🔴 **{status.key}** — failed")
        st.caption(status.error)
        return

    st.markdown(f"🟢 **{status.key}** — {len(status.exposed)}/{len(status.tools)} exposed")
    with st.expander("tools", expanded=False):
        for name in status.tools:
            mark = "✓" if name in status.exposed else "·"
            st.caption(f"{mark} {name.split(NAME_SEP, 1)[1]}")


def render_sidebar(host: MCPHost, presets: dict[str, str]) -> str | None:
    """
    Draw the server inventory, the tool budget, and the controls.

    Returns a preset prompt if one was clicked, otherwise None.
    """
    queued: str | None = None

    with st.sidebar:
        st.subheader("MCP servers")
        for status in host.status:
            _render_server_row(status)

        connected = sum(1 for s in host.status if s.connected)
        sent = len(to_groq_tools(host.tools, host.exposed_names))
        st.divider()
        st.caption(f"{connected}/{len(host.status)} connected · {len(host.tools)} discovered")
        st.caption(f"{sent} sent to `{config.MODEL}`")

        st.divider()
        st.subheader("Workspace")
        files = sorted(p.name for p in config.WORKSPACE_DIR.glob("*") if p.is_file())
        st.caption(", ".join(files) if files else "empty")

        st.divider()
        st.subheader("Try one")
        for label, text in presets.items():
            if st.button(label, width="stretch"):
                queued = text

        st.divider()
        if st.button("Reconnect", width="stretch"):
            # Order matters: stopping first terminates the five child processes.
            # Clearing the cache alone would leak them.
            host.stop()
            st.cache_resource.clear()
            st.rerun()

    return queued
