# query.py
import os
from pathlib import Path

from google import genai
from google.genai import types

import config

os.environ["CHROMA_TELEMETRY"] = "false"
import chromadb

def embed_query(client: genai.Client, question: str) -> list[float]:
    """Convert a user question into a vector for semantic search."""
    response = client.models.embed_content(
        model=config.EMBED_MODEL,
        contents=question,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=config.EMBED_DIM,
        ),
    )
    return response.embeddings[0].values

def retrieve_frames(
    collection: chromadb.Collection,
    query_vector: list[float],
) -> list[dict]:
    """Search ChromaDB for the frames most semantically similar to the question."""
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=config.TOP_K_RESULTS,
        include=["documents", "metadatas", "distances"],
    )

    frames = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        frames.append({
            "description": doc,
            "timestamp_str": meta["timestamp_str"],
            "frame_path": meta["frame_path"],
            "similarity": 1 - dist,
        })

    frames.sort(key=lambda x: x["timestamp_str"])
    return frames

def answer_question(
    client: genai.Client,
    question: str,
    frames: list[dict],
) -> str:
    """Send the retrieved frame images to Gemini and get a visual answer."""
    contents = []

    context_prompt = (
        f"You are answering a question about a video. "
        f"You have been given {len(frames)} frames retrieved from the video "
        f"that are most relevant to the question. "
        f"Each frame is labeled with its timestamp. "
        f"Answer the question based on what you can see in these frames."
    )
    contents.append(types.Part.from_text(text=context_prompt))

    for frame in frames:
        contents.append(
            types.Part.from_text(text=f"\n[Frame at {frame['timestamp_str']}]")
        )
        with open(frame["frame_path"], "rb") as f:
            img_bytes = f.read()
        contents.append(
            types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
        )

    contents.append(types.Part.from_text(text=f"\nQuestion: {question}"))

    response = client.models.generate_content(
        model=config.VISION_MODEL,
        contents=contents,
    )

    return response.text.strip()

def ask(
    client: genai.Client,
    collection: chromadb.Collection,
    question: str,
) -> str:
    """
    Single entry point for the CLI: embed question → retrieve frames → answer.
    Hides the three-step pipeline behind one clean call.
    """
    query_vector = embed_query(client, question)
    frames = retrieve_frames(collection, query_vector)

    if not frames:
        return "No relevant frames found for that question."

    return answer_question(client, question, frames)

