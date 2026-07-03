"""Central configuration and the three model functions RAG-Anything needs.

This module is the only place that talks to Gemini and to the local embedding
model. It exposes:

- ``llm_model_func``     -> text completion via Gemini (OpenAI-compatible endpoint)
- ``vision_model_func``  -> multimodal completion via Gemini (handles images)
- ``embedding_func``     -> local BAAI/bge-m3 embeddings (no API, no quota)
- ``get_rag()``          -> a lazily-built, reused RAGAnything instance

Everything is loaded once at module import (the embedding model) or once on
first use (the RAGAnything instance) so nothing heavy is rebuilt per call.

Inputs come only from environment variables (loaded from ``.env``). No keys are
hardcoded.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc
from raganything import RAGAnything, RAGAnythingConfig
from raganything import modalprocessors as _modalprocessors
from sentence_transformers import SentenceTransformer

# openai's exception types are what Gemini's OpenAI-compatible endpoint raises.
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

load_dotenv()  # must run before we read any key or build any client

logger = logging.getLogger("rag_anything_app")

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR: Path = BASE_DIR / "documents"
WORKING_DIR: Path = BASE_DIR / "rag_storage"
OUTPUT_DIR: Path = BASE_DIR / "output"

# --------------------------------------------------------------------------- #
# Gemini (OpenAI-compatible endpoint) settings
# --------------------------------------------------------------------------- #
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
GEMINI_BASE_URL: str = os.getenv(
    "GEMINI_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai/",
)
# gemini-2.5-flash is free, fast, and vision-capable — a good default for the
# many calls LightRAG makes during ingestion. Override via env if you want the
# newer gemini-3.5-flash (heavier "thinking" model, uses more free quota).
LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-2.5-flash")
VISION_MODEL: str = os.getenv("VISION_MODEL", "gemini-2.5-flash")

# Retry/backoff for Gemini 429 (rate limit) and transient 5xx errors.
MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "5"))
_RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)

# --------------------------------------------------------------------------- #
# Local embedding model settings
# --------------------------------------------------------------------------- #
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM: int = 1024  # bge-m3's real dimension — must match EmbeddingFunc
EMBEDDING_MAX_TOKENS: int = 8192

# Default retrieval mode for text queries ("hybrid" blends local + global graph
# retrieval). Other valid LightRAG modes: local, global, naive, mix.
QUERY_MODE: str = os.getenv("QUERY_MODE", "hybrid")


def require_api_key() -> str:
    """Return the Gemini key or raise a clear error if it is missing.

    Returns:
        The GEMINI_API_KEY value.

    Raises:
        RuntimeError: if GEMINI_API_KEY is not set in the environment / .env.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your "
            "key from https://aistudio.google.com/apikey"
        )
    return GEMINI_API_KEY


# --------------------------------------------------------------------------- #
# Embedding model — loaded exactly once
# --------------------------------------------------------------------------- #
_embed_model: SentenceTransformer | None = None
_embed_lock = threading.Lock()


def get_embedding_model() -> SentenceTransformer:
    """Load BAAI/bge-m3 once (thread-safe) and reuse it for every call.

    LightRAG calls the embedding function from a pool of worker threads. Without
    the lock, several threads race to construct the SentenceTransformer on first
    use, which corrupts the load ("Cannot copy out of meta tensor"). The lock +
    double-check guarantees a single construction.

    Also asserts the model's real embedding dimension matches ``EMBEDDING_DIM``.
    A mismatch here is the classic silent-retrieval-failure bug, so we fail loud.

    Returns:
        A ready SentenceTransformer instance.
    """
    global _embed_model
    if _embed_model is None:
        with _embed_lock:
            if _embed_model is None:
                logger.info("Loading embedding model %s ...", EMBEDDING_MODEL)
                model = SentenceTransformer(EMBEDDING_MODEL)
                get_dim = getattr(
                    model,
                    "get_embedding_dimension",
                    model.get_sentence_embedding_dimension,
                )
                real_dim = get_dim()
                if real_dim != EMBEDDING_DIM:
                    raise ValueError(
                        f"Embedding dim mismatch: {EMBEDDING_MODEL} produces "
                        f"{real_dim} but EMBEDDING_DIM is {EMBEDDING_DIM}. Fix "
                        f"EMBEDDING_DIM or the model — retrieval breaks silently."
                    )
                _embed_model = model
    return _embed_model


def warmup_embedding_model() -> None:
    """Materialize the model weights in the main thread before workers run.

    bge-m3's weights are lazily materialized on the first forward pass. Doing
    that first pass here (single-threaded) means LightRAG's worker threads only
    ever run inference on already-materialized weights — no meta-tensor race.
    """
    get_embedding_model().encode(
        ["warmup"], normalize_embeddings=True, show_progress_bar=False
    )


def _encode(texts: list[str]) -> np.ndarray:
    """Encode a batch of texts to normalized vectors (runs in a worker thread)."""
    model = get_embedding_model()
    return model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


async def bge_m3_embed(texts: list[str]) -> np.ndarray:
    """Async embedding function LightRAG will await.

    sentence-transformers has no async API, so the CPU-bound encode runs in a
    thread to avoid blocking the event loop.

    Args:
        texts: List of strings to embed.

    Returns:
        A numpy array of shape (len(texts), EMBEDDING_DIM).
    """
    return await asyncio.to_thread(_encode, texts)


embedding_func = EmbeddingFunc(
    embedding_dim=EMBEDDING_DIM,
    max_token_size=EMBEDDING_MAX_TOKENS,
    func=bge_m3_embed,
)


# --------------------------------------------------------------------------- #
# Gemini completion with retry/backoff
# --------------------------------------------------------------------------- #
async def _complete(model: str, prompt: str, **kwargs: Any) -> str:
    """Call Gemini's chat endpoint with exponential backoff on 429 / 5xx.

    Args:
        model: Gemini model id (e.g. "gemini-2.5-flash").
        prompt: The user prompt (ignored when ``messages`` is passed in kwargs).
        **kwargs: Passed through to openai_complete_if_cache (system_prompt,
            history_messages, messages, etc.).

    Returns:
        The model's text response.
    """
    api_key = require_api_key()
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return await openai_complete_if_cache(
                model,
                prompt,
                api_key=api_key,
                base_url=GEMINI_BASE_URL,
                **kwargs,
            )
        except _RETRYABLE as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                break
            wait = min(2**attempt, 30)  # 2, 4, 8, 16, 30... seconds
            logger.warning(
                "Gemini call failed (%s), retry %d/%d in %ds",
                type(exc).__name__,
                attempt,
                MAX_RETRIES,
                wait,
            )
            await asyncio.sleep(wait)
    raise RuntimeError(f"Gemini call failed after {MAX_RETRIES} retries") from last_error


def llm_model_func(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> Any:
    """Text completion function for LightRAG (returns an awaitable coroutine)."""
    return _complete(
        LLM_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        **kwargs,
    )


def vision_model_func(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    image_data: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> Any:
    """Multimodal completion function for RAG-Anything.

    Handles the three paths RAG-Anything uses:
    1. ``messages`` given  -> a pre-built VLM message list (VLM-enhanced query).
    2. ``image_data`` given -> a single base64 image to describe.
    3. neither             -> pure text, delegate to ``llm_model_func``.

    The Gemini OpenAI-compatible endpoint accepts the base64 ``image_url``
    data-URL format used below (verified against the live endpoint).
    """
    if messages:
        return _complete(VISION_MODEL, "", messages=messages, **kwargs)

    if image_data:
        return _complete(
            VISION_MODEL,
            "",
            messages=[
                {"role": "system", "content": system_prompt or ""},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            },
                        },
                    ],
                },
            ],
            **kwargs,
        )

    return llm_model_func(prompt, system_prompt, history_messages, **kwargs)


# --------------------------------------------------------------------------- #
# Compatibility shim for raganything 1.3.1 + this lightrag
# --------------------------------------------------------------------------- #
def _inject_role_llm_funcs(lightrag: Any) -> None:
    """Make ``role_llm_funcs`` a real attribute on the LightRAG instance.

    lightrag's ``extract_entities`` / ``merge_nodes_and_edges`` read
    ``global_config["role_llm_funcs"]["extract"]``. But that dict is normally
    only injected by ``lightrag._build_global_config()`` at call time — it is not
    a dataclass field. raganything 1.3.1's multimodal batch merge instead passes
    ``self.lightrag.__dict__`` straight through, so the key is missing and
    ingesting a document's figures/tables raises KeyError('role_llm_funcs').

    ``role_llm_funcs`` is a read-only property on LightRAG, so we write straight
    into ``__dict__``. Attribute access still goes through the property (a data
    descriptor wins over ``__dict__``), but ``__dict__["role_llm_funcs"]`` — the
    dict-subscript path raganything's batch merge uses — now resolves.
    """
    if lightrag is None or not hasattr(lightrag, "_build_global_config"):
        return
    gc = lightrag._build_global_config()
    lightrag.__dict__["role_llm_funcs"] = gc["role_llm_funcs"]
    lightrag.__dict__["llm_cache_identities"] = gc.get("llm_cache_identities")


def _patch_raganything_role_funcs() -> None:
    """Ensure ``role_llm_funcs`` is present wherever raganything reads it.

    Two code paths in raganything 1.3.1 drop the runtime ``role_llm_funcs``:
    the modal processors build ``global_config`` from ``asdict(lightrag)``, and
    the batch merge passes ``lightrag.__dict__``. We patch both:

    - each modal processor __init__ -> rebuild global_config via
      ``_build_global_config()``;
    - RAGAnything._ensure_lightrag_initialized -> stash ``role_llm_funcs`` as a
      real attribute on the lightrag instance (so ``__dict__`` carries it).
    """
    base = _modalprocessors.BaseModalProcessor
    for cls in [base, *base.__subclasses__()]:
        if "__init__" not in cls.__dict__ or "_role_funcs_patched" in cls.__dict__:
            continue
        original_init = cls.__init__

        def make_patched(orig):
            def patched_init(self, lightrag, *args, **kwargs):
                orig(self, lightrag, *args, **kwargs)
                if hasattr(lightrag, "_build_global_config"):
                    self.global_config = lightrag._build_global_config()

            return patched_init

        cls.__init__ = make_patched(original_init)
        cls._role_funcs_patched = True

    if not getattr(RAGAnything, "_role_funcs_patched", False):
        original_ensure = RAGAnything._ensure_lightrag_initialized

        async def patched_ensure(self):
            result = await original_ensure(self)
            _inject_role_llm_funcs(getattr(self, "lightrag", None))
            return result

        RAGAnything._ensure_lightrag_initialized = patched_ensure
        RAGAnything._role_funcs_patched = True


_patch_raganything_role_funcs()


# --------------------------------------------------------------------------- #
# RAGAnything instance — built once, reused everywhere
# --------------------------------------------------------------------------- #
_rag: RAGAnything | None = None


def build_config() -> RAGAnythingConfig:
    """Build the RAGAnythingConfig with all multimodal processing enabled."""
    return RAGAnythingConfig(
        working_dir=str(WORKING_DIR),
        parser="mineru",
        parse_method="auto",
        parser_output_dir=str(OUTPUT_DIR),
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
        recursive_folder_processing=True,
    )


async def get_rag() -> RAGAnything:
    """Return a lazily-built, shared RAGAnything instance.

    Also ensures the internal LightRAG instance is initialized (loading any
    existing graph from ``rag_storage/`` on disk). Plain-text ``aquery()``
    requires this to already be done — unlike the multimodal query methods, it
    does not lazily initialize LightRAG itself, so a bare query in a fresh
    process would otherwise fail with "No LightRAG instance available" even
    though documents were ingested in an earlier run.

    Returns:
        A RAGAnything wired to Gemini (text + vision) and local bge-m3 embeddings,
        with its LightRAG instance ready to query.
    """
    global _rag
    if _rag is None:
        require_api_key()
        warmup_embedding_model()  # single-threaded materialization before workers
        _rag = RAGAnything(
            config=build_config(),
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
        )
        logger.info("RAGAnything ready (LLM=%s, vision=%s)", LLM_MODEL, VISION_MODEL)
    if _rag.lightrag is None:
        init_result = await _rag._ensure_lightrag_initialized()
        if not init_result or not init_result.get("success"):
            raise RuntimeError(
                f"LightRAG initialization failed: "
                f"{(init_result or {}).get('error', 'unknown error')}"
            )
    return _rag


async def finalize() -> None:
    """Flush and close all storages cleanly (call once before the loop exits).

    LightRAG otherwise finalizes at interpreter exit, after the event loop is
    already closed, which spams "bound to a different event loop" errors and can
    drop the last vector batch. Calling this explicitly avoids that.
    """
    global _rag
    if _rag is not None:
        await _rag.finalize_storages()
