"""A semantic cache: reuse an answer when a new question means the same thing.

Inputs: questions (embedded locally by all-MiniLM-L6-v2, the ONNX build Chroma ships).
Outputs: a Hit with the cached answer and its similarity, or None on a miss.
Entries carry the policy version and a timestamp, so stale ones can be dropped.
"""

import time
import uuid
from dataclasses import dataclass

import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

from src.config import CACHE_DIR, COLLECTION, THRESHOLD, TTL_SECONDS

# One embedder for the whole process. Chroma's default embedding function
# reloads the ONNX model on every call (~0.2-1.6 s); this instance loads it once.
EMBEDDER = ONNXMiniLM_L6_V2()
# It also pads every question to 256 tokens. Mean pooling ignores padding, so
# turning it off gives the same vector ~10x faster. Embed one text per call.
EMBEDDER.tokenizer.no_padding()


@dataclass
class Hit:
    """A cached answer that is close enough to reuse."""

    answer: str
    question: str
    similarity: float
    age_s: float


class SemanticCache:
    """Question -> answer store, looked up by meaning instead of exact text."""

    def __init__(self, threshold: float = THRESHOLD, ttl_s: float = TTL_SECONDS) -> None:
        """Open (or create) the cache on disk."""
        self.threshold = threshold
        self.ttl_s = ttl_s
        client = chromadb.PersistentClient(path=str(CACHE_DIR))
        self.col = client.get_or_create_collection(
            COLLECTION,
            embedding_function=EMBEDDER,
            configuration={"hnsw": {"space": "cosine"}},
        )

    def lookup(self, question: str, policy_version: int) -> Hit | None:
        """Return the closest fresh entry if it clears the threshold, else None."""
        if self.col.count() == 0:
            return None
        result = self.col.query(
            query_texts=[question],
            n_results=1,
            where={"policy_version": policy_version},
        )
        if not result["ids"][0]:
            return None

        similarity = 1 - result["distances"][0][0]
        meta = result["metadatas"][0][0]
        age_s = time.time() - meta["created_at"]
        if age_s > self.ttl_s:
            # Expired: evict it so the fresh answer stored next takes its place.
            self.col.delete(ids=result["ids"][0])
            return None
        if similarity < self.threshold:
            return None
        return Hit(meta["answer"], result["documents"][0][0], similarity, age_s)

    def store(self, question: str, answer: str, policy_version: int) -> None:
        """Save a fresh answer under the question it was generated for."""
        self.col.add(
            ids=[uuid.uuid4().hex],
            documents=[question],
            metadatas=[{
                "answer": answer,
                "policy_version": policy_version,
                "created_at": time.time(),
            }],
        )

    def invalidate_similar(self, topic: str, min_similarity: float = 0.5) -> list[str]:
        """Delete every entry whose question is close to `topic`; return those questions."""
        if self.col.count() == 0:
            return []
        result = self.col.query(query_texts=[topic], n_results=self.col.count())
        doomed = [
            (entry_id, question)
            for entry_id, question, distance in zip(
                result["ids"][0], result["documents"][0], result["distances"][0]
            )
            if 1 - distance >= min_similarity
        ]
        if doomed:
            self.col.delete(ids=[entry_id for entry_id, _ in doomed])
        return [question for _, question in doomed]

    def clear(self) -> None:
        """Remove every entry."""
        ids = self.col.get()["ids"]
        if ids:
            self.col.delete(ids=ids)
