"""Writer Agent — an A2A-compliant server that rewrites text in a punchy style.

Input:  a research summary sent by another agent over A2A (JSON-RPC).
Output: a Task whose artifact holds a punchy, beginner-friendly rewrite.

Run:  PYTHONPATH=. uv run uvicorn agents.writer_agent:app --port 8002
Card: http://localhost:8002/.well-known/agent-card.json
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
    "You are an explainer. Rewrite the given text as a punchy, beginner-friendly "
    "explanation a smart 15-year-old would enjoy. Keep it to 3-4 short sentences. "
    "Use plain words and one concrete analogy. No preamble."
)


def build_agent_card() -> AgentCard:
    """Build this agent's public Agent Card (its discovery document)."""
    skill = AgentSkill(
        id="rewrite_simple",
        name="Rewrite Simply",
        description="Rewrites any text into a punchy, beginner-friendly explanation.",
        tags=["writing", "explainer"],
        examples=["Rewrite this dense paragraph so a beginner gets it"],
    )
    return AgentCard(
        name="Writer Agent",
        description="Rewrites text into a punchy, beginner-friendly explanation using Groq.",
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=False),
        supported_interfaces=[
            AgentInterface(url="http://localhost:8002", protocol_binding="JSONRPC"),
        ],
        skills=[skill],
    )


class WriterAgentExecutor(AgentExecutor):
    """Core logic: read incoming text, rewrite it punchy, publish as an artifact."""

    def __init__(self) -> None:
        self.groq = AsyncGroq()  # reads GROQ_API_KEY from the environment

    async def _rewrite(self, text: str) -> str:
        """Call Groq and return the rewritten text."""
        response = await self.groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            max_tokens=512,
        )
        return response.choices[0].message.content

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Handle one A2A request: text in, punchy rewrite out."""
        # 1. Get or create the task, enqueueing the Task object first.
        task = context.current_task or new_task_from_user_message(context.message)
        if not context.current_task:
            await event_queue.enqueue_event(task)

        # 2. Wrap the queue so we emit standard lifecycle events.
        updater = TaskUpdater(
            event_queue=event_queue, task_id=task.id, context_id=task.context_id
        )
        await updater.start_work(message=new_text_message("Rewriting..."))

        # 3. Do the actual work.
        rewrite = await self._rewrite(context.get_user_input())

        # 4. Attach the result as an artifact, then mark the task complete.
        await updater.add_artifact(parts=[new_text_part(rewrite, media_type="text/plain")])
        await updater.complete(message=new_text_message("Rewrite complete."))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """No long-running work to interrupt in this project."""
        return None


def build_app() -> Starlette:
    """Wire the executor + card into a Starlette app with A2A routes."""
    card = build_agent_card()
    handler = DefaultRequestHandler(
        agent_executor=WriterAgentExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    routes = [
        *create_agent_card_routes(card),
        *create_jsonrpc_routes(handler, "/"),
    ]
    return Starlette(routes=routes)


app = build_app()
