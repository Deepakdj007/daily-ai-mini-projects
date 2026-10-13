"""The LLM judge that watches retrieval, and the answer call that sits after it.

Inputs:  a query and the candidate passages the retriever returned
Outputs: one verdict per passage (answers / partial / unrelated + reason), and a grounded answer

The judge never sees similarity scores or ranks, and the passages arrive in
a shuffled order. If it saw the retriever's ranking it could simply agree
with it, and a judge that agrees with the retriever cannot debug it.
"""

import asyncio
import hashlib
import json
from dataclasses import dataclass, field

from groq import AsyncGroq, BadRequestError

from src import config
from src.retriever import Candidate

VERDICTS = ["answers", "partial", "unrelated"]

JUDGE_PROMPT = """You check the output of a search system for an online store.
You get a customer QUESTION and a list of PASSAGES from the store's help pages.

Label each passage:
  answers    read on its own, the passage gives the customer a correct, direct answer
  partial    same topic, but it leaves out or gets wrong the part the question asks about
  unrelated  a different topic

Judge meaning, not shared words. A passage can repeat the question's words
and still be about a different case. A passage can use none of them and
still be the answer.

reason: one short clause a developer can read to see why you chose the label."""

ANSWER_PROMPT = """Answer the customer's question using ONLY the passages given.
If they do not answer it, say so plainly and do not guess. Two or three sentences."""


@dataclass(frozen=True, slots=True)
class Verdict:
    """The judge's opinion of one passage."""

    label: str
    reason: str


@dataclass
class JudgeResult:
    """Verdicts keyed by passage id, plus what the call cost and whether it failed."""

    verdicts: dict[str, Verdict] = field(default_factory=dict)
    tokens: int = 0
    error: str = ""


_client: AsyncGroq | None = None


def client() -> AsyncGroq:
    """One shared async Groq client."""
    global _client
    if _client is None:
        _client = AsyncGroq(api_key=config.require_api_key())
    return _client


def shuffled_labels(candidates: list[Candidate]) -> dict[str, Candidate]:
    """Give each passage a neutral label (c1, c2...) in an order unrelated to its rank."""
    ordered = sorted(candidates, key=lambda c: hashlib.sha256(c.pid.encode()).hexdigest())
    return {f"c{i}": c for i, c in enumerate(ordered, start=1)}


def verdict_schema(labels: list[str]) -> dict:
    """A strict schema whose id field only accepts the labels we actually sent."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["verdicts"],
        "properties": {
            "verdicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "label", "reason"],
                    "properties": {
                        "id": {"type": "string", "enum": labels},
                        "label": {"type": "string", "enum": VERDICTS},
                        "reason": {"type": "string"},
                    },
                },
            }
        },
    }


async def _call_judge(messages: list[dict], schema: dict) -> tuple[dict, int]:
    """Send one strict-schema request, retrying once if gpt-oss breaks the schema."""
    for attempt in range(2):
        try:
            response = await client().chat.completions.create(
                model=config.JUDGE_MODEL,
                messages=messages,
                temperature=0.0,
                reasoning_effort="low",
                max_completion_tokens=config.JUDGE_MAX_TOKENS,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "verdicts", "strict": True, "schema": schema},
                },
            )
        except BadRequestError as exc:
            if "json_validate_failed" in str(exc) and attempt == 0:
                continue
            raise
        return json.loads(response.choices[0].message.content), response.usage.total_tokens
    raise RuntimeError("unreachable")


async def judge_pool(query: str, candidates: list[Candidate]) -> JudgeResult:
    """Label every candidate in one call. A failure comes back as JudgeResult.error, never raised."""
    labelled = shuffled_labels(candidates)
    listing = "\n".join(f"[{label}] {c.text}" for label, c in labelled.items())
    messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": f"QUESTION: {query}\n\nPASSAGES:\n{listing}"},
    ]
    try:
        data, tokens = await _call_judge(messages, verdict_schema(list(labelled)))
    except Exception as exc:  # the debugger must still show the retrieval half
        return JudgeResult(error=f"{type(exc).__name__}: {exc}"[:300])

    verdicts = {
        labelled[item["id"]].pid: Verdict(item["label"], item["reason"])
        for item in data["verdicts"]
        if item["id"] in labelled
    }
    return JudgeResult(verdicts=verdicts, tokens=tokens)


async def answer(query: str, kept: list[Candidate]) -> tuple[str, int]:
    """What the pipeline would tell the customer, from the kept passages only."""
    if not kept:
        return "(no passage survived retrieval, so there is nothing to answer from)", 0
    context = "\n".join(f"- {c.text}" for c in kept)
    response = await client().chat.completions.create(
        model=config.ANSWER_MODEL,
        messages=[
            {"role": "system", "content": ANSWER_PROMPT},
            {"role": "user", "content": f"PASSAGES:\n{context}\n\nQUESTION: {query}"},
        ],
        temperature=0.0,
        reasoning_effort="low",
        max_completion_tokens=config.ANSWER_MAX_TOKENS,
    )
    return (response.choices[0].message.content or "").strip(), response.usage.total_tokens


if __name__ == "__main__":
    right = Candidate("t#0", "t", "Orders can be cancelled free before dispatch, but not after.", 0.0, 1)
    wrong = Candidate("t#1", "t", "Brass diyas tarnish naturally; lemon and salt restore the shine.", 0.0, 2)
    result = asyncio.run(judge_pool("Can I cancel after dispatch?", [right, wrong]))
    assert not result.error, result.error
    print(result.verdicts, f"{result.tokens} tokens")
    assert result.verdicts["t#0"].label == "answers" and result.verdicts["t#1"].label == "unrelated"
    print("OK - the judge separates an answer from an unrelated passage")
