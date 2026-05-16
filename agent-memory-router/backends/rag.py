"""
backends/rag.py
---------------
Simplified RAG backend using ChromaDB (in-memory mode).
In a real system, this would connect to Qdrant, Pinecone, or pgvector
and use a proper embedding model. Here we use ChromaDB's default
embeddings for clarity.
"""

import chromadb

# Initialise in-memory ChromaDB client
_client = chromadb.Client()
_collection = _client.get_or_create_collection("documents")

# Seed the collection with example documents
_collection.add(
    documents=[
        "The refund policy allows returns within 30 days for electronics purchased online.",
        "All subscription plans include a 7-day free trial with no credit card required.",
        "For physical products, returns must be initiated via the support portal.",
    ],
    ids=["doc1", "doc2", "doc3"]
)

def query_rag(user_query: str, n_results: int = 2) -> str:
    """
    Queries the ChromaDB collection and returns the top matching documents
    as a single string, suitable for injection into an LLM context.
    """
    results = _collection.query(
        query_texts=[user_query],
        n_results=n_results
    )

    # Flatten the nested list ChromaDB returns
    docs = results["documents"][0]
    return "\n\n".join(docs)