"""The retriever under test: a Chroma collection searched by cosine similarity.

Inputs:  data/kumkum_home.json and a query string
Outputs: a ranked list of Candidate passages with their similarity scores

This file is deliberately ordinary. It is the "black box" the debugger
opens up, so it should look like the retriever in any RAG tutorial.
"""

import json
from dataclasses import dataclass

import chromadb

from src import config


@dataclass(frozen=True, slots=True)
class Candidate:
    """One passage the retriever returned, and where it ranked."""

    pid: str
    doc_id: str
    text: str
    similarity: float
    rank: int


def load_passages() -> list[dict]:
    """Flatten the corpus into one record per passage, with a stable id."""
    corpus = json.loads(config.CORPUS_PATH.read_text(encoding="utf-8"))
    passages = []
    for doc in corpus["docs"]:
        for index, text in enumerate(doc["passages"]):
            passages.append({"pid": f"{doc['doc_id']}#{index}", "doc_id": doc["doc_id"], "text": text})
    return passages


_collection = None


def collection():
    """Build the in-memory collection once per process and reuse it."""
    global _collection
    if _collection is None:
        client = chromadb.EphemeralClient()
        _collection = client.get_or_create_collection(
            name="kumkum_home",
            configuration={"hnsw": {"space": "cosine"}},
        )
        passages = load_passages()
        _collection.add(
            ids=[p["pid"] for p in passages],
            documents=[p["text"] for p in passages],
            metadatas=[{"doc_id": p["doc_id"]} for p in passages],
        )
    return _collection


def search(query: str, n: int) -> list[Candidate]:
    """Return the n nearest passages. Chroma gives cosine distance; similarity is 1 - distance."""
    result = collection().query(query_texts=[query], n_results=n)
    return [
        Candidate(
            pid=pid,
            doc_id=meta["doc_id"],
            text=text,
            similarity=round(1.0 - distance, 4),
            rank=rank,
        )
        for rank, (pid, text, meta, distance) in enumerate(
            zip(result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]),
            start=1,
        )
    ]


if __name__ == "__main__":
    queries = json.loads(config.QUERIES_PATH.read_text(encoding="utf-8"))
    print(f"indexed {collection().count()} passages")
    for item in queries:
        print(f"\n{item['label']}: {item['query']}")
        for c in search(item["query"], config.JUDGE_POOL):
            gold = "  <- answer" if c.pid in item["answer_pids"] else ""
            print(f"  {c.rank:>2}  {c.similarity:.3f}  {c.pid}{gold}")
