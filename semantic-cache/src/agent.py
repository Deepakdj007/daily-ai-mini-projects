"""The support agent: check the semantic cache first, call the LLM only on a miss.

Inputs: a question, the cache, and the current policy (text + version).
Outputs: a Record describing what happened - hit or miss, cost, latency.
"""

import time
from dataclasses import dataclass

from src.cache import SemanticCache
from src.llm import answer


@dataclass
class Record:
    """One answered question and what it cost."""

    question: str
    answer: str
    hit: bool
    cost: float
    latency_ms: float
    similarity: float = 0.0
    matched: str = ""


async def ask(
    question: str,
    policy: str,
    version: int,
    cache: SemanticCache | None,
) -> Record:
    """Answer one question, from the cache when possible. Pass cache=None to skip it."""
    lookup_ms = 0.0
    if cache is not None:
        start = time.perf_counter()
        hit = cache.lookup(question, version)
        lookup_ms = (time.perf_counter() - start) * 1000
        if hit is not None:
            return Record(question, hit.answer, True, 0.0, lookup_ms,
                          hit.similarity, hit.question)

    reply = await answer(question, policy)
    if cache is not None:
        cache.store(question, reply.text, version)
    # A miss pays for the lookup AND the model call.
    return Record(question, reply.text, False, reply.cost, lookup_ms + reply.latency_ms)
