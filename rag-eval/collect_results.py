# collect_results.py
# Runs the RAG pipeline on each test question and saves the outputs to JSON.

from dotenv import load_dotenv
load_dotenv()  # always first, before any other imports

import json
from rag_pipeline import build_vector_store, rag_answer


# ── Test dataset ──────────────────────────────────────────────────────────────
# Each entry has a question and a human-written reference answer.
# Reference answers are used by ContextPrecision and ContextRecall.
# Faithfulness and AnswerRelevancy don't need them.
TEST_QUESTIONS = [
    {
        "question": "What is RAG and why does it reduce hallucinations?",
        "reference": (
            "RAG stands for Retrieval-Augmented Generation. It retrieves relevant "
            "documents from an external database before generating an answer, "
            "grounding the response in real content. This reduces hallucinations "
            "because the LLM answers from retrieved documents rather than memory."
        ),
    },
    {
        "question": "What is ChromaDB and how does it find relevant documents?",
        "reference": (
            "ChromaDB is an open-source vector database. It stores text as embedding "
            "vectors and finds the documents whose embeddings are most similar to "
            "the query embedding, even without exact word matches."
        ),
    },
    {
        "question": "What does Faithfulness measure in RAGAS?",
        "reference": (
            "Faithfulness measures whether every claim in the generated answer "
            "can be traced back to the retrieved context. An unfaithful answer "
            "contains hallucinations — claims not supported by what was retrieved."
        ),
    },
    {
        "question": "How is AnswerRelevancy different from Faithfulness?",
        "reference": (
            "Faithfulness checks whether the answer is grounded in the retrieved "
            "context. AnswerRelevancy checks whether the answer actually addresses "
            "the user's question. An answer can be faithful but still score low "
            "on relevancy if it answers the wrong question."
        ),
    },
    {
        "question": "What changed in RAGAS v0.4 for initializing the evaluator LLM?",
        "reference": (
            "In RAGAS v0.4, the evaluator LLM is initialized with llm_factory from "
            "ragas.llms, which uses the google-genai client for "
            "Gemini. LangchainLLMWrapper is rejected at runtime by v0.4 metrics."
        ),
    },
    {
        "question": "What embedding model is used for the RAG pipeline and does it need an API key?",
        "reference": (
            "The all-MiniLM-L6-v2 model from HuggingFace is used. "
            "It runs entirely locally with no API key needed."
        ),
    },
]


def load_corpus(filepath: str) -> list[str]:
    """
    Reads the document corpus and splits it into individual paragraphs.

    Args:
        filepath: Path to the plain text corpus file.

    Returns:
        List of non-empty paragraph strings.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    return [p.strip() for p in content.split("\n\n") if p.strip()]


def collect_rag_outputs() -> list[dict]:
    """
    Runs the RAG pipeline on all test questions and captures outputs.

    For each question, captures:
    - The generated answer from the RAG pipeline
    - The retrieved context chunks (as plain strings)
    - The human-written reference answer

    Returns:
        List of dicts with keys: question, answer, contexts, reference.
    """
    print("Building vector store...")
    corpus = load_corpus("data/docs.txt")
    vector_store = build_vector_store(corpus)
    print(f"Indexed {len(corpus)} documents.\n")

    results = []
    for i, item in enumerate(TEST_QUESTIONS):
        question = item["question"]
        reference = item["reference"]
        print(f"[{i+1}/{len(TEST_QUESTIONS)}] {question[:65]}...")

        answer, contexts = rag_answer(question, vector_store)

        results.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,      # list of retrieved chunk strings
            "reference": reference,    # human-written ground truth
        })
        print(f"  Answer: {answer[:80]}...\n")

    return results


if __name__ == "__main__":
    results = collect_rag_outputs()
    with open("rag_outputs.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} results to rag_outputs.json")