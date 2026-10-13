"""The retrieval evaluator: grades each retrieved passage, then picks one CRAG action.

Inputs:  the question and the retrieved Docs
Outputs: a label per passage and a verdict - CORRECT, AMBIGUOUS or INCORRECT
"""

import asyncio

from src.kb import Doc, retrieve
from src.llm import ask_json

LABELS = ["relevant", "partial", "irrelevant"]

SYSTEM = """You grade passages retrieved for a railway help desk question.
For each passage choose one label:
- relevant: the passage alone fully answers the question.
- partial: the passage covers part of the question but leaves something out.
- irrelevant: the passage does not help answer this question, even if it shares words with it.
Judge each passage on its own. Do not answer the question."""

SCHEMA = {
    "type": "object",
    "properties": {
        "grades": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string", "enum": LABELS},
                },
                "required": ["id", "label"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["grades"],
    "additionalProperties": False,
}


async def grade(question: str, docs: list[Doc]) -> dict[str, str]:
    """Label every passage in one batched call. Returns {passage id: label}."""
    passages = "\n\n".join(f"[{d.id}]\n{d.text}" for d in docs)
    user = f"Question: {question}\n\nPassages:\n\n{passages}"
    reply = await ask_json(SYSTEM, user, "grades", SCHEMA)
    labels = {g["id"]: g["label"] for g in reply["grades"]}
    # A passage the model skipped counts as irrelevant rather than crashing the run.
    return {d.id: labels.get(d.id, "irrelevant") for d in docs}


def decide(labels: dict[str, str]) -> str:
    """Turn passage labels into one of the three CRAG actions."""
    values = list(labels.values())
    if "relevant" in values:
        return "CORRECT"
    if all(v == "irrelevant" for v in values):
        return "INCORRECT"
    return "AMBIGUOUS"


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "What time does Tatkal booking open for AC classes?"
    docs = retrieve(question)
    labels = asyncio.run(grade(question, docs))
    for d in docs:
        print(f"  {d.similarity:.2f}  {labels[d.id]:<10}  {d.id}")
    print(f"verdict: {decide(labels)}")
