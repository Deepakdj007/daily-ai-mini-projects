"""Local MiniLM embeddings, exposed as the plain callable SqliteStore wants.

LangGraph's index config accepts `Callable[[Sequence[str]], list[list[float]]]`
directly, so there is no need for a LangChain Embeddings wrapper here.

The model is loaded lazily behind a lock. Streamlit reruns the script on every
interaction and can touch this from more than one thread; without the lock, two
threads race to build the same SentenceTransformer and one of them can observe a
half-initialised model.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence

from src.config import DIMS, EMBED_MODEL

_model = None
_lock = threading.Lock()


def _get_model():
    """Load the sentence-transformers model once, on first use."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:  # re-check: another thread may have won the race
                from sentence_transformers import SentenceTransformer

                _model = SentenceTransformer(EMBED_MODEL)
    return _model


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Embed a batch of strings into 384-dimension vectors.

    Args:
        texts: strings to embed.

    Returns:
        One list of floats per input string, in the same order.
    """
    model = _get_model()
    vectors = model.encode(list(texts), normalize_embeddings=True)
    return [v.tolist() for v in vectors]


def warm_up() -> None:
    """Force the model download/load now instead of mid-conversation."""
    embed_texts(["warm up"])


if __name__ == "__main__":
    vecs = embed_texts(["webhooks stopped firing", "settlement is late"])
    print(f"model      : {EMBED_MODEL}")
    print(f"vectors    : {len(vecs)}")
    print(f"dimensions : {len(vecs[0])} (config says {DIMS})")
    assert len(vecs[0]) == DIMS, f"DIMS mismatch: model gives {len(vecs[0])}"
    print("OK")
