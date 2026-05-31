"""Research Agent — an A2A-compliant server that summarizes a topic with Groq.

Input:  a plain-text topic sent by another agent over A2A (JSON-RPC).
Output: a Task whose artifact holds a 3-4 sentence research summary.

Run:  PYTHONPATH=. uv run uvicorn agents.research_agent:app --port 8001
Card: http://localhost:8001/.well-known/agent-card.json
"""

from dotenv import load_dotenv
from groq import AsyncGroq
from starlette.applications import Starlette

from a2a.helpers import (
    new_task_from_user_message,
    new_text_message,
    new_text_part,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill

# Load GROQ_API_KEY before any AsyncGroq() is constructed (mandatory on Windows).
load_dotenv()

SYSTEM_PROMPT = (
    "You are a research assistant. Given a topic, write a tight, factual "
    "summary in 3-4 sentences. No preamble, no bullet points."
)


def build_agent_card() -> AgentCard:
    """Build this agent's public Agent Card (its discovery document)."""
    skill = AgentSkill(
        id="summarize_topic",
        name="Summarize Topic",
        description="Generates a concise 3-4 sentence research summary of any topic.",
        tags=["research", "summary"],
        examples=["Summarize transformers in AI"],
    )
    return AgentCard(
        name="Research Agent",
        description="Summarizes any topic into a short research brief using Groq.",
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=False),
        supported_interfaces=[
            AgentInterface(url="http://localhost:8001", protocol_binding="JSONRPC"),
        ],
        skills=[skill],
    )


class ResearchAgentExecutor(AgentExecutor):
    """Core logic: read the topic, call Groq, publish a summary artifact."""

    def __init__(self) -> None:
        self.groq = AsyncGroq()  # reads GROQ_API_KEY from the environment

    async def _summarize(self, topic: str) -> str:
        """Call Groq and return the summary text."""
        response = await self.groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": topic},
            ],
            max_tokens=512,
        )
        return response.choices[0].message.content

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Handle one A2A request: topic in, summary artifact out."""
        # 1. Get the existing task or create one. The Task MUST be enqueued
        #    before any status update, or the SDK raises InvalidAgentResponseError.
        task = context.current_task or new_task_from_user_message(context.message)
        if not context.current_task:
            await event_queue.enqueue_event(task)

        # 2. TaskUpdater wraps the queue so we emit standard lifecycle events.
        updater = TaskUpdater(
            event_queue=event_queue, task_id=task.id, context_id=task.context_id
        )
        await updater.start_work(message=new_text_message("Researching the topic..."))

        # 3. Do the actual work.
        summary = await self._summarize(context.get_user_input())

        # 4. Attach the result as an artifact, then mark the task complete.
        await updater.add_artifact(parts=[new_text_part(summary, media_type="text/plain")])
        await updater.complete(message=new_text_message("Research complete."))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """No long-running work to interrupt in this project."""
        return None


def build_app() -> Starlette:
    """Wire the executor + card into a Starlette app with A2A routes."""
    card = build_agent_card()
    handler = DefaultRequestHandler(
        agent_executor=ResearchAgentExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    routes = [
        *create_agent_card_routes(card),
        *create_jsonrpc_routes(handler, "/"),
    ]
    return Starlette(routes=routes)


app = build_app()
