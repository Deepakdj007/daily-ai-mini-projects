"""Knowledge refinement: break passages into sentence strips and keep only the useful ones.

Inputs:  the question and the retrieved Docs
Outputs: a short text made only of the strips that help answer the question
"""

import asyncio
import re

from src.kb import Doc, retrieve
from src.llm import ask_json

SYSTEM = """You filter sentences for a railway help desk.
For each numbered sentence, set keep to true only if it helps answer the question.
Drop sentences that are on a nearby topic but do not help with this question."""

SCHEMA = {
    "type": "object",
    "properties": {
        "strips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "keep": {"type": "boolean"}},
                "required": ["id", "keep"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["strips"],
    "additionalProperties": False,
}


def split_strips(docs: list[Doc]) -> dict[str, str]:
    """Split every passage into sentences, numbered s1, s2, ... across all passages."""
    strips: dict[str, str] = {}
    for doc in docs:
        for sentence in re.split(r"(?<=[.!?])\s+", doc.text):
            if sentence.strip():
                strips[f"s{len(strips) + 1}"] = f"{sentence.strip()} [KB:{doc.id}]"
    return strips


async def refine(question: str, docs: list[Doc]) -> list[str]:
    """Return only the strips the model marks as useful, in their original order."""
    strips = split_strips(docs)
    numbered = "\n".join(f"{sid}: {text}" for sid, text in strips.items())
    reply = await ask_json(SYSTEM, f"Question: {question}\n\n{numbered}", "strips", SCHEMA)
    kept = {s["id"] for s in reply["strips"] if s["keep"]}
    return [text for sid, text in strips.items() if sid in kept]


if __name__ == "__main__":
    question = "What time does Tatkal booking open for AC classes?"
    docs = retrieve(question)
    print(f"{len(split_strips(docs))} strips in, kept:")
    for strip in asyncio.run(refine(question, docs)):
        print(f"  {strip}")
