"""Qdrant multivector store — holds one multivector per page and ranks by MaxSim.

A normal vector DB stores one vector per item. ColPali gives us *many* vectors
per page (one per patch), so the collection is configured as a multivector with
the MAX_SIM comparator: for each query token Qdrant takes its best-matching page
patch, then sums those bests. That late-interaction score is what finds the page
whose visual regions answer the question.

Runs fully local and on-disk (no server, no Docker) via QdrantClient(path=...).
"""

from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from src.config import COLLECTION_NAME, QDRANT_PATH, TOP_K, VECTOR_DIM

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    """Open the on-disk Qdrant store once and reuse the connection."""
    global _client
    if _client is None:
        _client = QdrantClient(path=str(QDRANT_PATH))
    return _client


def close_client() -> None:
    """Close the local store explicitly so the file lock is released cleanly.

    Without this, Qdrant's __del__ runs at interpreter shutdown and fails to
    import portalocker (sys.meta_path is gone), printing a noisy traceback.
    """
    global _client
    if _client is not None:
        _client.close()
        _client = None


def ensure_collection(reset: bool = False) -> None:
    """Create the multivector collection if it is missing.

    Args:
        reset: When True, drop any existing collection first (fresh ingest).
    """
    client = get_client()
    if reset and client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qm.VectorParams(
                size=VECTOR_DIM,
                distance=qm.Distance.COSINE,
                multivector_config=qm.MultiVectorConfig(
                    comparator=qm.MultiVectorComparator.MAX_SIM
                ),
            ),
        )


def upsert_page(
    point_id: int,
    multivector: list[list[float]],
    pdf_name: str,
    page_number: int,
    image_path: Path,
) -> None:
    """Store one page's multivector plus the metadata we need at answer time."""
    client = get_client()
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            qm.PointStruct(
                id=point_id,
                vector=multivector,
                payload={
                    "pdf": pdf_name,
                    "page_number": page_number,
                    "image_path": str(image_path),
                },
            )
        ],
    )


def search(query_multivector: list[list[float]], top_k: int = TOP_K) -> list[dict]:
    """Return the top-k pages for a query multivector, best score first."""
    client = get_client()
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_multivector,
        limit=top_k,
        with_payload=True,
    )
    return [
        {**point.payload, "score": round(point.score, 4)}
        for point in response.points
    ]
