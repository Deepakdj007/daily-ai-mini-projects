"""Gradio chat UI over the ingested documents.

A single-page chat: type a question, get an answer. A checkbox toggles
VLM-enhanced mode, which re-reads relevant figures through the vision model for
richer answers (uses more free vision quota, so it is off by default).

Run with:  PYTHONPATH=. uv run python -m app.main ui
"""

from __future__ import annotations

import logging

import gradio as gr

from .chat import ask, ask_vlm_enhanced

logger = logging.getLogger("rag_anything_app")


async def _respond(message: str, history: list, vlm_enhanced: bool) -> str:
    """Answer one chat turn.

    Args:
        message: The user's question.
        history: Gradio chat history (unused; retrieval is over the graph).
        vlm_enhanced: If True, route retrieved images through the vision model.

    Returns:
        The answer text (or a readable error string).
    """
    message = (message or "").strip()
    if not message:
        return "Ask a question about your documents."
    try:
        if vlm_enhanced:
            return await ask_vlm_enhanced(message)
        return await ask(message)
    except Exception as exc:  # noqa: BLE001 - surface errors in the UI, don't crash
        logger.exception("Query failed")
        return f"Error: {exc}"


def build_ui() -> gr.Blocks:
    """Build the Gradio Blocks app."""
    with gr.Blocks(title="RAG-Anything Document Chat") as demo:
        gr.Markdown(
            "# 📄 RAG-Anything Document Chat\n"
            "Ask questions about the documents you ingested "
            "(`PYTHONPATH=. uv run python -m app.main ingest ./documents`)."
        )
        vlm_toggle = gr.Checkbox(
            label="VLM-enhanced mode (re-reads figures — uses more vision quota)",
            value=False,
        )
        gr.ChatInterface(
            fn=_respond,
            additional_inputs=[vlm_toggle],
            # Each example is [message, vlm_enhanced] because we have an
            # additional input (the VLM toggle).
            examples=[
                ["What is this document about?", False],
                ["Summarize the key findings.", False],
                ["What does Figure 1 show?", True],
            ],
        )
    return demo


def launch() -> None:
    """Launch the Gradio server on http://localhost:7860."""
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860)
