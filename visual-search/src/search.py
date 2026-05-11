"""
search.py
---------
Provides search_products(): the core text-to-image search function.
Called from the CLI test entry point and from the FastAPI endpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedder import CLIPEmbedder
from src.store import VectorStore

# These module-level variables hold our shared instances.
# They start as None and are created on the first call to their getter function.
_embedder: CLIPEmbedder | None = None
_store: VectorStore | None = None

def _get_embedder() -> CLIPEmbedder:
    """
    Return the shared CLIPEmbedder, creating it on the first call.

    This is the Singleton pattern. The CLIP model is ~350MB.
    If we created a new CLIPEmbedder() inside search_products() directly,
    every single search request would reload 350MB from disk — adding
    5–10 seconds of latency per search. The singleton ensures we pay
    that loading cost exactly once per process lifetime.
    """
    global _embedder
    if _embedder is None:
        _embedder = CLIPEmbedder()
    return _embedder


def _get_store() -> VectorStore:
    """Return the shared VectorStore, creating it on the first call."""
    global _store
    if _store is None:
        _store = VectorStore()
    return _store

def search_products(query: str, top_k: int = 5) -> list[dict]:
    """
    Find product images that best match a plain-English text query.

    Args:
        query: Natural language description, e.g. "black formal shoes".
        top_k: Number of results to return (default 5).

    Returns:
        List of result dicts, sorted by descending similarity score.
        Each dict has: id, score, image_path, filename, category.

    Raises:
        ValueError: If query is empty or only whitespace.
    """
    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    embedder = _get_embedder()
    store = _get_store()

    # Step 1: Encode the text query to a 512-dim vector.
    # This is the only time CLIP runs during a search. It is fast (~50ms on CPU)
    # because text encoding is much lighter than image encoding.
    query_vector = embedder.embed_text(query)

    # Step 2: Search Qdrant for the closest image vectors.
    # Qdrant computes cosine similarity between the query_vector and all
    # 500 stored image vectors, then returns the top_k highest scores.
    raw_results = store.search(query_vector, top_k=top_k)

    # Step 3: Flatten the nested payload into a clean flat dict.
    # raw_results look like: [{"id": 3, "score": 0.28, "payload": {"image_path": ..., ...}}]
    # We move the payload fields to the top level for easier use by callers.
    return [
        {
            "id": r["id"],
            "score": r["score"],
            "image_path": r["payload"].get("image_path", ""),
            "filename": r["payload"].get("filename", ""),
            "category": r["payload"].get("category", "unknown"),
        }
        for r in raw_results
    ]

# This block runs ONLY when you execute this file directly:
#     uv run python src/search.py "sneakers"
#
# When search.py is imported by main.py, this block is skipped entirely.
# sys.argv[1] reads the first command-line argument after the script name.
if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "sneakers"
    print(f"\n🔍 Searching for: '{query}'\n")

    results = search_products(query, top_k=5)

    for rank, r in enumerate(results, start=1):
        print(
            f"  #{rank}  "
            f"score={r['score']:.4f}  "
            f"category={r['category']:<12}  "   # :<12 left-pads to 12 chars for alignment
            f"file={r['filename']}"
        )