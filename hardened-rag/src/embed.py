"""Local MiniLM embeddings, as a plain callable.

Inputs:  a sequence of strings
Outputs: a list of 384-float unit vectors

Free and local, so the ablation can re-embed a whole store on every arm without
touching a quota. The model is loaded once behind a double-checked lock because
Streamlit reruns the script on a different thread and two threads racing to
load a transformer is a slow way to run out of memory.

Vectors are L2-normalised, which is what lets sqlite-vec's cosine distance be
read as a similarity with `1 - distance`.
"""

from __future__ import annotations

import threading
from typing import Sequence

from src import config

_model = None
_lock = threading.Lock()


def _get_model():
    """Load MiniLM once per process."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:  # re-check: another thread may have won the race
                from sentence_transformers import SentenceTransformer

                _model = SentenceTransformer(config.EMBED_MODEL)
    return _model


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Embed a batch. Returns one unit vector per input, in order."""
    if not texts:
        return []
    vectors = _get_model().encode(list(texts), normalize_embeddings=True)
    return [vector.tolist() for vector in vectors]


def embed_one(text: str) -> list[float]:
    """Embed a single string."""
    return embed_texts([text])[0]


def warm_up() -> None:
    """Pay the load cost now rather than inside a user-visible action."""
    _get_model()


if __name__ == "__main__":
    vectors = embed_texts(["Lives in Pune.", "Lives in Bengaluru.", "Drives a Honda City."])
    assert len(vectors) == 3
    assert len(vectors[0]) == config.DIMS, f"expected {config.DIMS}, got {len(vectors[0])}"

    norm = sum(value * value for value in vectors[0]) ** 0.5
    assert abs(norm - 1.0) < 1e-4, f"vectors must be unit length, got {norm}"

    def cosine(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    city_pair = cosine(vectors[0], vectors[1])
    unrelated = cosine(vectors[0], vectors[2])
    assert city_pair > unrelated, "two city facts must be closer than a city and a car"
    print(f"OK - {config.DIMS}d unit vectors; city/city {city_pair:.3f} > city/car {unrelated:.3f}")
