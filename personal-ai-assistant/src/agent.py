"""Build the tool-calling agent and give it a safe way to be invoked from a
Telegram handler — Groq failures become a friendly reply, not a crash.

Inputs:  a ChatGroq instance, the tool list, and a LangGraph checkpointer.
Outputs: build_agent() returns the compiled graph; ask() invokes it for one chat.
"""

import logging

from langchain.agents import create_agent
from langchain.agents.middleware import ClearToolUsesEdit, ContextEditingMiddleware
from langchain_core.messages import HumanMessage

from .prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Some free-tier Groq orgs cap tokens-per-minute as low as 8000 — far below the
# model's 131k context window. Tool outputs (search results, article/PDF text)
# are what actually fills up a long-running conversation's history, so we clear
# old ones out well before that cap instead of letting every turn grow forever.
_CONTEXT_EDITING = ContextEditingMiddleware(
    edits=[ClearToolUsesEdit(trigger=3000, keep=2, clear_tool_inputs=True)],
)


def build_agent(llm, tools: list, checkpointer):
    """Wire the Groq model, the tool list, and the sqlite checkpointer together."""
    return create_agent(
        llm,
        tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
        middleware=[_CONTEXT_EDITING],
    )


async def ask(agent, chat_id: int, text: str) -> str:
    """Run one turn for a chat. thread_id=str(chat_id) is what gives each chat
    its own durable conversation memory via the sqlite checkpointer."""
    run_config = {"configurable": {"thread_id": str(chat_id)}}
    try:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=text)]}, config=run_config
        )
    except Exception as exc:
        logger.exception("agent invocation failed")
        message = str(exc)
        if "tool_use_failed" in message:
            return "I got tangled up trying to use a tool for that — could you rephrase?"
        if "429" in message or "rate_limit" in message.lower():
            return "I've hit my free Groq request limit for now — try again in a bit."
        return "Something went wrong on my end handling that. Try again?"
    return result["messages"][-1].content
