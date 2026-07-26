"""The reasoning core: given one changed doc, propose grounded FAQ edits.

Inputs:  an AsyncGroq client, the changed doc (path + text), and the FAQ entries
         currently grounded in that doc.
Outputs: a validated list of Operation objects (add / update / remove), each with a
         citation anchor and a plain-English reason for the change.

We use Groq's json_schema structured output in STRICT mode. gpt-oss is a reasoning
model: the ordinary tool-calling structured-output path makes it emit prose, which Groq
rejects with `400 tool_use_failed`. A json_schema response_format constrains the raw
output to exactly this shape instead.

Two reliability tweaks for that reasoning model. First, a generous max_completion_tokens:
gpt-oss spends tokens on reasoning before the JSON, and if the budget runs out mid-reason
it returns an empty generation that fails strict validation with `400 json_validate_failed`.
Second, an escalating retry. Because temperature=0 is deterministic, a plain retry would
fail identically, so each retry changes the generation conditions: first add a little
temperature to break the loop, then fall back to best-effort (non-strict) validation.
"""

import json
import sqlite3
from typing import Literal

from groq import AsyncGroq, BadRequestError
from pydantic import BaseModel

from src.config import MODEL
from src.watcher import DocEvent


class Operation(BaseModel):
    """One proposed edit to the FAQ.

    op:            'add', 'update', or 'remove'.
    faq_id:        the existing FAQ id for update/remove; -1 for add.
    question:      the FAQ question (echo the existing one for remove).
    answer:        the grounded answer ('' for remove).
    source_anchor: the exact source heading the answer is drawn from (the citation).
    reason:        why this edit fired — the drift it responds to.
    """

    op: Literal["add", "update", "remove"]
    faq_id: int
    question: str
    answer: str
    source_anchor: str
    reason: str


class OperationBatch(BaseModel):
    """The full set of edits the agent wants to make for one changed doc."""

    operations: list[Operation]


# Hand-written schema: strict mode requires every property in `required` and
# `additionalProperties: false` on every object. Pydantic's generated schema does
# not add the latter, so we spell it out here.
_OPERATION_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["operations"],
    "properties": {
        "operations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["op", "faq_id", "question", "answer", "source_anchor", "reason"],
                "properties": {
                    "op": {"type": "string", "enum": ["add", "update", "remove"]},
                    "faq_id": {"type": "integer"},
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                    "source_anchor": {"type": "string"},
                    "reason": {"type": "string"},
                },
            },
        }
    },
}

_SYSTEM = (
    "You are FAQ Autopilot, a long-horizon agent that keeps a customer-facing FAQ "
    "accurate and fully grounded in a set of source documents. You are given ONE "
    "source document that just changed, plus the FAQ entries currently grounded in "
    "it. Decide the MINIMAL set of edits that keeps the FAQ correct.\n\n"
    "Rules:\n"
    "- Every answer must be supported by THIS document. Never invent facts.\n"
    "- 'update' an existing entry whose answer is now wrong or outdated. Reuse its faq_id.\n"
    "- 'add' a new entry only for a high-value question the document now supports and "
    "that the existing entries do not already cover. Use faq_id -1.\n"
    "- 'remove' an existing entry the document no longer supports. Reuse its faq_id.\n"
    "- Leave correct entries alone: do NOT emit an operation for them.\n"
    "- source_anchor must be the exact heading text the answer comes from (e.g. "
    "'Refund policy'). This is the citation.\n"
    "- Keep answers to 1-3 sentences. reason must state the specific drift you saw.\n"
    "- Aim for 2-4 well-chosen questions per document, not an exhaustive list."
)


def _build_user_prompt(doc: DocEvent, existing: list[sqlite3.Row]) -> str:
    """Assemble the user message: the changed doc plus the FAQ entries citing it."""
    existing_json = json.dumps(
        [
            {"faq_id": row["id"], "question": row["question"], "answer": row["answer"]}
            for row in existing
        ],
        indent=2,
    )
    return (
        f"SOURCE DOCUMENT: {doc.rel_path} (event: {doc.kind})\n"
        f"------------------------------------------------------------\n"
        f"{doc.content}\n"
        f"------------------------------------------------------------\n\n"
        f"EXISTING FAQ ENTRIES GROUNDED IN THIS DOCUMENT (may be empty):\n"
        f"{existing_json}\n\n"
        f"Return the minimal set of operations to keep the FAQ correct and grounded."
    )


# Each attempt changes the generation conditions so a deterministic failure can't
# repeat forever: (temperature, strict). Attempt 1 is the fast deterministic path.
_ATTEMPTS: tuple[tuple[float, bool], ...] = ((0.0, True), (0.4, True), (0.4, False))


def _response_format(strict: bool) -> dict:
    """Build the json_schema response_format, strict or best-effort."""
    return {
        "type": "json_schema",
        "json_schema": {"name": "faq_operations", "strict": strict, "schema": _OPERATION_SCHEMA},
    }


async def propose_operations(
    client: AsyncGroq, doc: DocEvent, existing: list[sqlite3.Row]
) -> list[Operation]:
    """Ask the model for the FAQ edits a changed doc calls for.

    Args:
      client:   an AsyncGroq client.
      doc:      the added/modified doc (its .content is the new text).
      existing: active FAQ rows currently grounded in this doc.
    Returns:
      A list of validated Operation objects (possibly empty).
    """
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _build_user_prompt(doc, existing)},
    ]
    last_error: Exception | None = None
    for temperature, strict in _ATTEMPTS:
        try:
            response = await client.chat.completions.create(
                model=MODEL,
                temperature=temperature,
                reasoning_effort="medium",
                max_completion_tokens=8192,
                messages=messages,
                response_format=_response_format(strict),
            )
            raw = response.choices[0].message.content or "{}"
            return OperationBatch.model_validate_json(raw).operations
        except BadRequestError as err:  # json_validate_failed / empty generation
            last_error = err
    raise RuntimeError(
        f"Agent could not produce valid operations for {doc.rel_path} after "
        f"{len(_ATTEMPTS)} attempts: {last_error}"
    )
