"""The desk agent — a LangGraph loop of recall, answer, write.

Two persistence layers are wired here and they do different jobs:

- `SqliteSaver` (checkpointer) carries the conversation WITHIN one thread.
- `SqliteStore` (store) carries facts, episodes and the playbook ACROSS threads.

Every probe in the ablation harness runs on a fresh thread_id, so the
checkpointer has nothing to offer. Anything the agent recalls came from the
store — which is why a probe failure cannot be blamed on the context window.
"""

from __future__ import annotations

import sqlite3
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.config import get_store
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.store.sqlite import SqliteStore
from pydantic import BaseModel, Field

from src import prompt as prompt_mod
from src import semantic
from src.config import CHECKPOINT_PATH, MemoryConfig
from src.llm import invoke as llm_invoke


class Severity(str, Enum):
    """How urgent the ticket is."""

    low = "low"
    normal = "normal"
    high = "high"


class Triage(BaseModel):
    """The agent's answer plus the two decisions worth checking mechanically.

    `escalate` is a real decision, not a self-report, which is why the harness
    can check the escalation rule on a field instead of a regex. There is
    deliberately no `followed_the_playbook` flag — a model will happily claim it
    obeyed a rule it ignored.
    """

    reply: str = Field(description="what to send the customer")
    severity: Severity = Field(description="ticket severity")
    escalate: bool = Field(description="true if this must go to on-call now")


class DeskState(TypedDict):
    """Graph state. Every value here has to survive msgpack for the checkpointer."""

    messages: Annotated[list[AnyMessage], add_messages]
    customer_id: str
    system: str
    trace: dict[str, Any]
    reply: str
    severity: str
    escalate: bool


def _memory_from(config: RunnableConfig) -> MemoryConfig:
    """Read the ablation switches out of the run config.

    Passed through `configurable` rather than state so the three booleans never
    become part of the checkpointed conversation.
    """
    raw = (config.get("configurable") or {}).get("memory") or {}
    return MemoryConfig(
        semantic=raw.get("semantic", True),
        episodic=raw.get("episodic", True),
        procedural=raw.get("procedural", True),
    )


def recall(state: DeskState, config: RunnableConfig) -> dict[str, Any]:
    """Assemble the system prompt from whichever tiers are enabled."""
    store = get_store()
    ticket = state["messages"][-1].content
    system, trace = prompt_mod.build(
        store, state["customer_id"], ticket, _memory_from(config)
    )
    return {
        "system": system,
        "trace": {
            "summary": trace.summary(),
            "sizes": trace.sizes,
            "facts": [h.value["text"] for h in trace.facts],
            "fact_scores": [round(h.score or 0.0, 4) for h in trace.facts],
            "episodes": [h.value["text"] for h in trace.episodes],
            "episode_scores": [round(h.score or 0.0, 4) for h in trace.episodes],
            "rules": list(trace.playbook.rules) if trace.playbook else [],
            "system_chars": len(system),
        },
    }


def answer(state: DeskState) -> dict[str, Any]:
    """Produce the reply and the two triage decisions.

    Asks for the raw response alongside the parsed object so the real token
    counts come back from the API. Per-tier token counts are still reported as
    characters — splitting a total by tier would need gpt-oss's tokenizer, which
    is not available locally, and a different tokenizer would give a confidently
    wrong number.
    """
    messages = [("system", state["system"]), *state["messages"]]
    out = llm_invoke(Triage, messages, temperature=0.3, include_raw=True)
    result: Triage = out["parsed"]
    usage = getattr(out["raw"], "usage_metadata", None) or {}
    trace = dict(state.get("trace") or {})
    trace["tokens"] = {
        "input": usage.get("input_tokens", 0),
        "output": usage.get("output_tokens", 0),
    }
    return {
        "messages": [AIMessage(content=result.reply)],
        "reply": result.reply,
        "severity": result.severity.value,
        "escalate": result.escalate,
        "trace": trace,
    }


def write(state: DeskState, config: RunnableConfig) -> dict[str, Any]:
    """Extract durable account facts from the customer's turn and upsert them.

    Skipped when `learn` is false. The ablation harness runs against a pre-seeded
    store and ablates reads only, so it has no reason to pay for extraction.
    """
    if not (config.get("configurable") or {}).get("learn", True):
        return {}
    store = get_store()
    human = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    if not human:
        return {}
    facts = semantic.extract_facts(human[-1].content)
    written = semantic.write(store, state["customer_id"], facts)
    trace = dict(state.get("trace") or {})
    trace["written_facts"] = written
    return {"trace": trace}


def open_checkpointer(path: Path | str = CHECKPOINT_PATH) -> SqliteSaver:
    """Open the thread-history checkpointer.

    Direct construction again — `SqliteSaver.from_conn_string` closes its
    connection on exit, which is wrong for a long-lived process.
    """
    conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


def build_graph(store: SqliteStore, checkpointer: SqliteSaver | None = None):
    """Compile the desk graph with long-term memory attached.

    Args:
        store: long-term memory, injected so nodes reach it via `get_store()`.
        checkpointer: thread history. Optional so tests can skip it.

    Returns:
        The compiled graph.
    """
    builder = StateGraph(DeskState)
    builder.add_node("recall", recall)
    builder.add_node("answer", answer)
    builder.add_node("write", write)
    builder.add_edge(START, "recall")
    builder.add_edge("recall", "answer")
    builder.add_edge("answer", "write")
    builder.add_edge("write", END)
    return builder.compile(store=store, checkpointer=checkpointer)


def ask_desk(
    graph,
    customer_id: str,
    text: str,
    thread_id: str,
    memory: MemoryConfig | None = None,
    learn: bool = True,
) -> DeskState:
    """Send one message to the desk.

    Args:
        graph: compiled graph from `build_graph`.
        customer_id: who is writing.
        text: their message.
        thread_id: the conversation. A fresh id means no thread history at all.
        memory: which tiers may be read. Defaults to all three.
        learn: whether to run the semantic write step.

    Returns:
        The final state, including `reply` and `trace`.
    """
    memory = memory or MemoryConfig()
    config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
            "memory": {
                "semantic": memory.semantic,
                "episodic": memory.episodic,
                "procedural": memory.procedural,
            },
            "learn": learn,
        }
    }
    return graph.invoke(
        {"messages": [HumanMessage(content=text)], "customer_id": customer_id},
        config=config,
    )
