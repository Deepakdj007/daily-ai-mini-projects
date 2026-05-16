"""
router.py
---------
Uses Groq LLM to classify a query into one of three memory types:
    - rag      → semantic / fuzzy search over documents
    - graph    → relational / multi-hop entity reasoning
    - tabular  → exact structured lookup (numbers, records, IDs)

Returns a lowercase string: "rag", "graph", or "tabular"
"""

import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

ROUTER_SYSTEM_PROMPT = """
You are a memory routing classifier for an AI agent.

Given a user query, your only job is to classify which memory backend should handle it.

Memory backends:
- rag      → Use when the query needs semantic search over unstructured text documents.
             Example: "What does the refund policy say about electronics?"
- graph    → Use when the query needs to trace relationships or connections between entities.
             Example: "How is Project X related to the Q3 slowdown?"
- tabular  → Use when the query needs an exact, deterministic lookup from structured records.
             Example: "What did user 42 order last month?"

Respond with ONLY one word: rag, graph, or tabular.
No explanation. No punctuation. Just the word.
"""

def classify_query(query: str) -> str:
    """
    Classifies the query and returns the memory backend to use.
    Returns one of: "rag", "graph", "tabular"
    """
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ],
        max_tokens=10,
        temperature=0.0,  # Deterministic classification
    )

    # Strip whitespace and lowercase for safety
    label = response.choices[0].message.content.strip().lower()

    # Fallback to RAG if the model returns something unexpected
    if label not in ("rag", "graph", "tabular"):
        print(f"[router] Unexpected label '{label}', defaulting to rag")
        return "rag"

    return label