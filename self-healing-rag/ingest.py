import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from dotenv import load_dotenv

load_dotenv()

# PersistentClient saves data to disk so we don't re-ingest every run
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Create (or get) our collection - think of this as a "table" in ChromaDB
collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    metadata={"hnsw:space": "cosine"}  # cosine similarity for text
)

def load_and_split_documents(docs_folder: str) -> list[dict]:
    """Load all .txt files and split into chunks for storage."""
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,       # max 300 characters per chunk
        chunk_overlap=50,     # 50-character overlap keeps context
        length_function=len,
    )

    docs_path = Path(docs_folder)
    for txt_file in docs_path.glob("*.txt"):
        loader = TextLoader(str(txt_file), encoding="utf-8")
        documents = loader.load()
        chunks = text_splitter.split_documents(documents)
        all_chunks.extend(chunks)
        print(f"Loaded {len(chunks)} chunks from {txt_file.name}")

    return all_chunks

def ingest_to_chromadb(chunks: list) -> None:
    """Store document chunks in ChromaDB with auto-generated embeddings."""
    documents = [chunk.page_content for chunk in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {"source": chunk.metadata.get("source", "unknown")}
        for chunk in chunks
    ]

    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB.")

if __name__ == "__main__":
    print("Starting document ingestion...")
    chunks = load_and_split_documents("docs")

    if not chunks:
        print("No documents found in docs/ folder. Add .txt files first.")
    else:
        ingest_to_chromadb(chunks)
        print(f"Ingestion complete! Total chunks stored: {len(chunks)}")