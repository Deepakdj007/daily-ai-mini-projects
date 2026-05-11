"""
store.py
--------
Manages a Qdrant vector collection for product image embeddings.

Uses local path mode: data is persisted to disk between Python runs.
To switch to Qdrant Cloud later, change exactly one line in __init__.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# The name of our collection — like a table name in a relational database.
# One Qdrant instance can hold many collections. We use one.
COLLECTION_NAME = "products"

# Must match the output dimension of the CLIP model we are using.
# clip-vit-base-patch32  → 512 dimensions
# clip-vit-large-patch14 → 768 dimensions (if you upgrade later)
VECTOR_DIM = 512

# Cosine distance measures the angle between two vectors.
# The correct choice when vectors are L2-normalized (which ours are).
DISTANCE = Distance.COSINE

# Where Qdrant saves its files on disk.
# Delete this folder to wipe the index and start fresh.
DB_PATH = "data/qdrant_db"

class VectorStore:
    """
    Wraps Qdrant with helpers for inserting and searching image vectors.
    """

    def __init__(self) -> None:
        # QdrantClient(path=DB_PATH) creates a local Qdrant database
        # that saves its files inside data/qdrant_db/.
        #
        # This is different from QdrantClient(":memory:") which stores
        # everything in RAM and loses all data when Python exits.
        #
        # We use path= mode so the index survives between runs —
        # index once, then start the API server as many times as you like.
        self.client = QdrantClient(path=DB_PATH)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the collection on first run; connect to it on subsequent runs."""

        # Get a list of all existing collection names in this Qdrant instance
        existing_names = [
            c.name for c in self.client.get_collections().collections
        ]

        if COLLECTION_NAME not in existing_names:
            # First run — create a new collection.
            # VectorParams tells Qdrant the shape of every vector we will store:
            #   size=VECTOR_DIM → each vector has 512 numbers
            #   distance=DISTANCE → use cosine similarity when searching
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=DISTANCE),
            )
            print(f"✅ Created Qdrant collection: '{COLLECTION_NAME}'")
        else:
            # Already exists from a previous run — just report how many are stored
            count = self.client.count(COLLECTION_NAME).count
            print(f"✅ Connected to '{COLLECTION_NAME}' ({count} vectors indexed)")


    def upsert_batch(
        self,
        ids: list[int],
        vectors: list[np.ndarray],
        payloads: list[dict[str, Any]],
    ) -> None:
        """
        Insert or update a batch of vectors with their metadata.

        'Upsert' = insert if the ID is new, update if the ID already exists.
        This makes the indexer idempotent — safe to re-run without duplicates.

        Args:
            ids:      Unique integer ID for each vector (its primary key).
            vectors:  List of 512-dim normalized numpy arrays.
            payloads: Metadata to store alongside each vector — image path, category, etc.
                      Returned with search results so we know which image matched.
        """
        # Build a list of PointStruct objects — one per image.
        # Each PointStruct is one row in the Qdrant collection.
        points = [
            PointStruct(
                id=point_id,
                vector=vec.tolist(),   # Qdrant requires a plain Python list, not numpy
                payload=meta,          # any JSON-serializable dict of metadata
            )
            for point_id, vec, meta in zip(ids, vectors, payloads)
        ]

        # Send the entire batch to Qdrant in a single call.
        # This is much faster than calling upsert() 500 separate times —
        # it reduces disk I/O from 500 individual writes to one batch write.
        self.client.upsert(collection_name=COLLECTION_NAME, points=points)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        """
        Find the top_k image vectors most similar to the query vector.

        Args:
            query_vector: 512-dim text embedding from CLIP (normalized).
            top_k:        How many results to return.

        Returns:
            List of dicts sorted by descending score (best match first).
            Each dict has: id, score, payload.
        """
        # query_points() is the current Qdrant search API.
        # Note: the older .search() method was deprecated in qdrant-client 1.10+.
        # If you see AttributeError on .search(), upgrade to .query_points().
        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector.tolist(),   # the vector to search against
            limit=top_k,                   # return at most this many results
        ).points

        # Reshape results into plain dicts for easier use downstream.
        # hit.score is the cosine similarity — ranges from 0.0 (no match) to 1.0 (identical).
        return [
            {
                "id": hit.id,
                "score": round(hit.score, 4),
                "payload": hit.payload,    # the metadata we stored during indexing
            }
            for hit in results
        ]

    def count(self) -> int:
        """Return the total number of vectors currently stored."""
        return self.client.count(COLLECTION_NAME).count