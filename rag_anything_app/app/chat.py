"""Query layer: ask the knowledge graph questions (text and multimodal).

Two entry points:
- ``ask``                -> plain text query over everything already ingested.
- ``ask_with_multimodal`` -> attach a table / equation / image to the question
                            so the model reasons over that content specifically.

Both are async and reuse the single shared RAGAnything instance.
"""

from __future__ import annotations

import logging
from typing import Any

from .config import QUERY_MODE, get_rag

logger = logging.getLogger("rag_anything_app")


async def ask(question: str, mode: str | None = None) -> str:
    """Answer a text question from the ingested knowledge graph.

    Figures and tables are already described into the knowledge graph at
    ingestion time, so plain text retrieval sees them. We pass
    ``vlm_enhanced=False`` to skip RAG-Anything's query-time image re-read
    (which also avoids a NoneType crash in that path on empty retrieval).

    Args:
        question: The user's question.
        mode: Retrieval mode (hybrid/local/global/naive/mix). Defaults to
            QUERY_MODE from config ("hybrid").

    Returns:
        The answer text.
    """
    rag = await get_rag()
    result = await rag.aquery(question, mode=mode or QUERY_MODE, vlm_enhanced=False)
    return result


async def ask_vlm_enhanced(question: str, mode: str | None = None) -> str:
    """Answer a text question but let the vision model re-read relevant images.

    RAG-Anything's ``vlm_enhanced`` flag routes retrieved image content back
    through ``vision_model_func`` for a richer answer. This uses more of the
    free vision quota, so it is opt-in.

    Args:
        question: The user's question.
        mode: Retrieval mode. Defaults to QUERY_MODE.

    Returns:
        The answer text.
    """
    rag = await get_rag()
    return await rag.aquery(question, mode=mode or QUERY_MODE, vlm_enhanced=True)


async def ask_with_multimodal(
    question: str,
    multimodal_content: list[dict[str, Any]],
    mode: str | None = None,
) -> str:
    """Answer a question with attached multimodal content (table/equation/image).

    Example ``multimodal_content`` for a table::

        [{
            "type": "table",
            "table_data": "Model,Accuracy\\nBGE-M3,0.91\\nMiniLM,0.86",
            "table_caption": "Retrieval accuracy",
        }]

    And for an equation::

        [{"type": "equation", "latex": "E = mc^2"}]

    Args:
        question: The user's question about the attached content.
        multimodal_content: List of content dicts RAG-Anything understands.
        mode: Retrieval mode. Defaults to QUERY_MODE.

    Returns:
        The answer text.
    """
    rag = await get_rag()
    return await rag.aquery_with_multimodal(
        question,
        multimodal_content=multimodal_content,
        mode=mode or QUERY_MODE,
    )
