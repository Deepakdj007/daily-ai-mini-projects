"""
Streamlit front end for the MCP host.

The sidebar is the point of the app: five independent servers, what each one
offers, and how much of it the model can actually see. The chat pane shows every
tool call routed to its server as it happens.

Inputs: a prompt from the chat box.
Outputs: rendered conversation, plus real files and notes on disk.

Run: PYTHONPATH=. uv run streamlit run src/app.py
"""

from __future__ import annotations

import streamlit as st

from src import config
from src.agent import ERROR, TEXT, TOOL_CALL, TOOL_RESULT, run_turn
from src.host import MCPHost
from src.ui import render_call, render_sidebar, render_transcript

st.set_page_config(page_title="MCP Host", page_icon="🔌", layout="wide")

PRESETS = {
    "All five servers": (
        "What time is it in Kolkata right now? Then fetch https://example.com and write "
        "a 3-bullet summary to mcp-summary.md. Save a note tagged 'research' about what "
        "you learned, and tell me the subject of the most recent commit in this repo."
    ),
    "Just one server": "What time is it in Tokyo, and how many hours ahead of Kolkata is that?",
    "Recall a note": "What notes do I have tagged 'research'? List the files in my workspace too.",
}


@st.cache_resource(show_spinner=False)
def get_host() -> MCPHost:
    """
    Boot the host once and keep it for the life of the Streamlit process.

    cache_resource is the lifetime anchor. Without it every rerun would spawn a
    fresh set of five subprocesses and orphan the previous ones.
    """
    host = MCPHost()
    host.start()
    return host


def handle_turn(host: MCPHost, prompt: str) -> None:
    """Run one user turn, rendering tool calls live as the agent works."""
    st.session_state.transcript.append({"role": "user", "content": prompt, "calls": []})
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    calls: list[dict] = []
    pending: dict[str, dict] = {}
    answer = ""

    with st.chat_message("assistant"):
        for kind, payload in run_turn(host, st.session_state.client, st.session_state.messages):
            if kind == TOOL_CALL:
                pending[payload["id"]] = dict(payload)
            elif kind == TOOL_RESULT:
                entry = pending.pop(payload["id"], dict(payload))
                entry["result"] = payload["text"]
                calls.append(entry)
                render_call(entry)
            elif kind == TEXT:
                answer = f"{answer}\n\n{payload}".strip()
                st.markdown(payload)
            elif kind == ERROR:
                st.warning(payload)

    st.session_state.transcript.append({"role": "assistant", "content": answer, "calls": calls})


def boot() -> MCPHost:
    """Check the key, connect the servers, and seed session state."""
    try:
        config.require_keys()
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    with st.spinner("Connecting to MCP servers (first run downloads them)…"):
        try:
            host = get_host()
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()

    st.session_state.setdefault("client", config.make_client())
    st.session_state.setdefault("transcript", [])
    st.session_state.setdefault("messages", [{"role": "system", "content": config.SYSTEM_PROMPT}])
    return host


def main() -> None:
    """Wire the page together."""
    st.title("🔌 MCP Host")
    st.caption("One prompt, five independent MCP servers, one tool namespace.")

    host = boot()
    queued = render_sidebar(host, PRESETS)
    render_transcript(st.session_state.transcript)

    prompt = queued or st.chat_input("Ask across all five servers…")
    if prompt:
        handle_turn(host, prompt)


main()
