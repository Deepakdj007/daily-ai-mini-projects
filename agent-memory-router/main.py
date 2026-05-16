"""
main.py
-------
Entry point for the agent memory router demo.
Routes each query to the correct backend, retrieves context,
then generates a final answer using Groq.

Run with:
    PYTHONPATH=. uv run main.py
"""

import os
from dotenv import load_dotenv
load_dotenv()

from groq import Groq
from router import classify_query
from backends.rag import query_rag
from backends.graph import query_graph
from backends.tabular import query_tabular

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

ANSWER_SYSTEM_PROMPT = """
You are a helpful AI assistant. Answer the user's question using ONLY the context provided.
If the context does not contain enough information, say so clearly.
Be concise and direct.
"""

def answer_with_context(query: str, context: str) -> str:
    """
    Sends the query + retrieved context to Groq and returns a final answer.
    """
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ],
        max_tokens=512,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def run(query: str) -> None:
    """
    Full pipeline: classify → retrieve → generate → print.
    """
    print(f"\n{'='*60}")
    print(f"Query: {query}")

    # Step 1: Classify
    memory_type = classify_query(query)
    print(f"→ Router classified as: [{memory_type.upper()}]")

    # Step 2: Retrieve from the right backend
    if memory_type == "rag":
        context = query_rag(query)
    elif memory_type == "graph":
        context = query_graph(query)
    else:
        context = query_tabular(query)

    print(f"→ Retrieved context:\n{context}")

    # Step 3: Generate final answer
    answer = answer_with_context(query, context)
    print(f"\n✅ Answer:\n{answer}")


if __name__ == "__main__":
    # Test queries — one for each memory type
    queries = [
        "What does the refund policy say about electronics?",       # → RAG
        "How did Project Apollo affect Q3 APAC Revenue?",            # → Graph
        "What did user 42 order and how much did they spend total?", # → Tabular
    ]

    for q in queries:
        run(q)