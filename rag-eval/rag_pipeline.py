# rag_pipeline.py
# A simple RAG pipeline using ChromaDB + Groq.
# This is the system we will evaluate with RAGAS.

# load_dotenv must be called before any LLM or embedding imports.
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def build_vector_store(texts: list[str]) -> Chroma:
    """
    Splits raw text into chunks and indexes them in an in-memory ChromaDB store.
    Uses all-MiniLM-L6-v2 — a fast local embedding model, no API key needed.

    Args:
        texts: List of raw text strings to index.

    Returns:
        A Chroma vector store ready to query.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,    # max characters per chunk
        chunk_overlap=50,  # overlap to preserve context at boundaries
    )
    docs = [Document(page_content=t) for t in texts]
    split_docs = splitter.split_documents(docs)

    # Runs locally — no API key needed
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # In-memory Chroma store — no persistence needed for evaluation
    return Chroma.from_documents(split_docs, embeddings)


def rag_answer(question: str, vector_store: Chroma) -> tuple[str, list[str]]:
    """
    Runs the RAG pipeline for a single question.

    Returns both the generated answer AND the retrieved context chunks,
    because RAGAS needs the contexts separately to evaluate faithfulness
    and context quality.

    Args:
        question: The user's question.
        vector_store: A built Chroma vector store.

    Returns:
        Tuple of (generated_answer, list_of_retrieved_context_strings).
    """
    # Retrieve the top 3 most semantically similar chunks
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    retrieved_docs = retriever.invoke(question)

    # RAGAS needs plain strings, not Document objects
    retrieved_contexts = [doc.page_content for doc in retrieved_docs]

    # Build the prompt with retrieved context stuffed in
    context_block = "\n\n".join(retrieved_contexts)
    prompt = f"""You are a helpful assistant. Answer the question using ONLY
the information provided in the context below. If the answer is not in the
context, say "I don't know."

Context:
{context_block}

Question: {question}

Answer:"""

    # model= is the correct parameter name for ChatGroq (not model_name=)
    llm = ChatGroq(model="llama-3.3-70b-versatile")
    response = llm.invoke(prompt)

    return response.content, retrieved_contexts