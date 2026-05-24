# src/config.py
# ─────────────────────────────────────────────────────────────
# mem0 configuration for the persistent memory agent.
# Wires together: Groq LLM, HuggingFace local embedder,
# and Qdrant vector store stored on local disk.
# All components are free — no credit card required.
# ─────────────────────────────────────────────────────────────

import os
from dotenv import load_dotenv

# Load .env before anything else.
# If this line comes after any mem0 import, the API key
# won't be available when mem0 initializes its clients.
load_dotenv()

LLM_CONFIG = {
    "provider": "groq",
    "config": {
        "model": "llama-3.3-70b-versatile",
        "temperature": 0.1,   # Low temp = consistent fact extraction
        "max_tokens": 2000,
    }
}

EMBEDDER_CONFIG = {
    "provider": "huggingface",
    "config": {
        "model": "multi-qa-MiniLM-L6-cos-v1",
        "embedding_dims": 384,   # This model outputs 384-dimensional vectors
    }
}

VECTOR_STORE_CONFIG = {
    "provider": "qdrant",
    "config": {
        "collection_name": "memory_agent",
        "path": "/tmp/qdrant",   # Stored on disk — persists between runs
        "embedding_model_dims": 384,  # Must match EMBEDDER_CONFIG above
    }
}

def get_config() -> dict:
    """
    Build and return the complete mem0 Memory configuration.

    mem0's Memory.from_config() accepts a flat dictionary with
    three top-level keys: 'llm', 'embedder', and 'vector_store'.
    Each key maps to a provider name and a provider-specific config dict.

    Returns:
        dict: Complete mem0-compatible configuration dictionary.
    """
    return {
        "llm": LLM_CONFIG,
        "embedder": EMBEDDER_CONFIG,
        "vector_store": VECTOR_STORE_CONFIG,
    }