"""One turn of conversation: remember, answer, and write back what was learned.

Inputs:  a message, a user, a clock, and the behaviour switches
Outputs: an answer plus whatever the store decided to do about the message

Plain async functions rather than a second graph. A turn never pauses for a
human, so checkpointing it would buy nothing and cost several rows per node per
turn. The one moment that does need durability - a repair the agent is unsure
about - reaches the graph as a leaf call, which also keeps the whole path
synchronous where SqliteSaver needs it to be.

The answer instruction is byte-identical in every configuration. Only the
memory block above it changes. If the instruction moved too, an arm would be
measuring how the question was framed as well as what the store remembered.
"""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass, field

from src import cache, classify, config, extract, llm, policy as policymod, recall, repair
from src.policy import RepairPlan, WritePolicy

ANSWER_PREFIX = "Answer:"
UNKNOWN = "UNKNOWN"

ANSWER_INSTRUCTION = (
    "Answer using the memories above and what the user has just said. Do not use "
    "anything else you happen to know.\n"
    f"Reply with exactly one line: {ANSWER_PREFIX} <answer>.\n"
    f"If neither the memories nor the message contains the answer, reply "
    f"{ANSWER_PREFIX} {UNKNOWN}.\n"
    "If the memory you need is marked NEEDS VERIFICATION, or says it may be out of "
    "date or has ended, do not state it as current fact. Give it with the caveat: "
    f"{ANSWER_PREFIX} <value> (unverified since <date>).\n"
    "Never invent a value that is not in the memories."
)


@dataclass(slots=True)
class TurnResult:
    """Everything one turn produced, for the CLI, the UI and the evaluation."""

    answer: str = ""
    hits: list = field(default_factory=list)
    memory_block: str = ""
    facts: list = field(default_factory=list)
    plans: list = field(default_factory=list)
    outcomes: list = field(default_factory=list)
    parked: list[str] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    replayed: bool = False
    error: str = ""


def build_messages(memory_block: str, message: str, today: str) -> list[dict]:
    """The answer prompt. Memory varies between arms; nothing else does."""
    system = (
        "You are a personal assistant with a long-term memory of this user.\n"
        f"Today is {today}.\n\n"
        f"What you remember about them:\n{memory_block}\n\n{ANSWER_INSTRUCTION}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": message}]


async def answer(conn: sqlite3.Connection, user_id: str, message: str, *, clock,
                 as_of: int | None = None, model: str = "", show_freshness: bool = True,
                 window: bool = True, use_memory: bool = True, k: int = 0) -> tuple[str, list, str]:
    """Recall, then answer. Returns (answer line, hits, the rendered block)."""
    at = as_of if as_of is not None else clock.now()
    hits = recall.recall(conn, user_id, message, clock=clock, as_of=at, k=k,
                         window=window) if use_memory else []
    block = recall.render(hits, show_freshness=show_freshness) if use_memory else "(no memories on file)"
    from src import clock as clockmod

    messages = build_messages(block, message, clockmod.to_iso(at))
    key = cache.key(model or config.CHAT_MODEL, {
        "kind": "answer", "prompt": config.PROMPT_VERSION,
        "prompt_hash": cache.fingerprint(ANSWER_INSTRUCTION),
        "block": block, "message": message, "today": clockmod.to_iso(at),
    })
    completion = await llm.complete(messages, model=model or config.CHAT_MODEL, cache_key=key,
                                    max_completion_tokens=config.MAX_COMPLETION_TOKENS["answer"])
    return completion.text, hits, block


async def ingest(conn: sqlite3.Connection, user_id: str, message: str, *, clock,
                 write_policy: WritePolicy, gate_enabled: bool = True, graph=None,
                 oracle_human=None, model: str = "", said_at: int | None = None) -> TurnResult:
    """Extract facts from a message and write them under the given policy."""
    at = said_at if said_at is not None else clock.now()
    result = TurnResult()

    extraction = await extract.extract(message, said_at=at, model=model)
    result.facts = extraction.facts
    result.replayed = extraction.replayed
    if extraction.error:
        result.error = extraction.error
        return result

    def _launch(store_conn, plan, *, clock, policy, oracle_human):
        from src.graph import launch_repair

        outcome = launch_repair(graph, store_conn, plan, clock=clock, policy=policy,
                                oracle_human=oracle_human)
        if outcome == "parked":
            result.parked.append(plan.topic)
        return outcome

    for fact in extraction.facts:
        candidates = classify.candidates_for(
            conn, user_id, fact, at=at, scope_aware=write_policy.scope_aware,
            window=write_policy.history,
        ) if write_policy.resolves else []
        relations: list = []
        if candidates:
            relations, error = await classify.classify(fact, candidates, model=model)
            if error:
                result.error = error
        for plan in policymod.plan(fact, candidates, relations,
                                   policy=write_policy, user_id=user_id):
            result.plans.append(plan)
            result.outcomes.append(repair.apply_or_park(
                conn, plan, clock=clock, policy=write_policy, gate_enabled=gate_enabled,
                launcher=_launch if graph is not None else None, oracle_human=oracle_human,
            ))
    return result


async def turn(conn: sqlite3.Connection, user_id: str, message: str, *, clock,
               write_policy: WritePolicy, gate_enabled: bool = True, graph=None,
               oracle_human=None, model: str = "", said_at: int | None = None,
               do_answer: bool = True, show_freshness: bool = True) -> TurnResult:
    """A full turn: answer the user and learn from what they said.

    The two halves both depend only on the message and the store as it was, so
    they run together rather than one after the other.
    """
    at = said_at if said_at is not None else clock.now()
    if do_answer:
        answered, ingested = await asyncio.gather(
            answer(conn, user_id, message, clock=clock, as_of=at, model=model,
                   show_freshness=show_freshness, window=write_policy.history),
            ingest(conn, user_id, message, clock=clock, write_policy=write_policy,
                   gate_enabled=gate_enabled, graph=graph, oracle_human=oracle_human,
                   model=model, said_at=at),
        )
        ingested.answer, ingested.hits, ingested.memory_block = answered
        return ingested
    return await ingest(conn, user_id, message, clock=clock, write_policy=write_policy,
                        gate_enabled=gate_enabled, graph=graph, oracle_human=oracle_human,
                        model=model, said_at=at)


if __name__ == "__main__":
    from src import clock as clockmod, store
    from src.policy import BITEMPORAL, OVERWRITE

    async def _smoke() -> None:
        for label, write_policy in (("bitemporal", BITEMPORAL), ("overwrite", OVERWRITE)):
            conn = store.connect(":memory:")
            clock = clockmod.FrozenClock("2026-06-01")
            await turn(conn, "u", "I live in Pune and I drive a Honda City.", clock=clock,
                       write_policy=write_policy, gate_enabled=False, do_answer=False)
            clock.advance(days=120)
            await turn(conn, "u", "I moved to Bengaluru last month.", clock=clock,
                       write_policy=write_policy, gate_enabled=False, do_answer=False)

            now = store.as_of(conn, "u", "city", "", clock.now())
            july = store.as_of(conn, "u", "city", "", clockmod.to_epoch("2026-07-01"))
            print(f"{label:11} now={now['value'] if now else None:>10}  "
                  f"July={july['value'] if july else None}")
            assert now and now["value"] == "Bengaluru", "both policies must get the present right"
            if write_policy.history:
                assert july and july["value"] == "Pune", "history must keep the past answerable"
            else:
                assert july is None, "overwrite loses the past entirely"
            assert not store.check_invariants(conn), store.check_invariants(conn)

        # And the answer path reads the block it was given.
        conn = store.connect(":memory:")
        clock = clockmod.FrozenClock("2026-06-01")
        await turn(conn, "u", "I live in Pune.", clock=clock, write_policy=BITEMPORAL,
                   gate_enabled=False, do_answer=False)
        result = await turn(conn, "u", "Which city do I live in?", clock=clock,
                            write_policy=BITEMPORAL, gate_enabled=False)
        assert "pune" in result.answer.lower(), result.answer
        print(f"answered    {result.answer.strip()!r}")

        empty = store.connect(":memory:")
        blank = await turn(empty, "u", "Which city do I live in?", clock=clock,
                           write_policy=BITEMPORAL, gate_enabled=False, do_answer=True)
        assert UNKNOWN in blank.answer.upper(), blank.answer
        print(f"no memory   {blank.answer.strip()!r}")
        print("OK - the same conversation, two policies, only one can still answer about July")

    asyncio.run(_smoke())
