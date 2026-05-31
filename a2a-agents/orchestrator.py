"""Orchestrator — the "agent hiring agents" flow over A2A.

Discovers the Research and Writer agents from their Agent Cards, sends a topic
to Research, takes its summary artifact, sends that to Writer, and prints the
punchy rewrite. Both agents must be running (ports 8001 and 8002).

Run: PYTHONPATH=. uv run python orchestrator.py
"""

import asyncio

import httpx

from a2a.client import A2ACardResolver, create_client
from a2a.helpers import get_artifact_text, new_text_message
from a2a.types import Role, SendMessageRequest

RESEARCH_URL = "http://localhost:8001"
WRITER_URL = "http://localhost:8002"
TOPIC = "Tell me about transformers in AI"


async def call_agent(http_client: httpx.AsyncClient, base_url: str, text: str) -> str:
    """Discover the agent at base_url, send it text, return its artifact text."""
    # 1. Discovery: fetch the Agent Card from /.well-known/agent-card.json
    resolver = A2ACardResolver(httpx_client=http_client, base_url=base_url)
    card = await resolver.get_agent_card()
    print(f"  -> discovered '{card.name}' at {base_url}")

    # 2. Build a client from the card. It reads supported_interfaces to find the
    #    JSON-RPC endpoint URL itself — we never hardcode the endpoint.
    client = await create_client(agent=card)

    # 3. Build the request. role MUST be ROLE_USER; messageId is auto-generated.
    request = SendMessageRequest(message=new_text_message(text, role=Role.ROLE_USER))

    # 4. send_message is an async generator. A non-streaming agent yields one
    #    StreamResponse carrying the final Task.
    final_task = None
    async for response in client.send_message(request):
        if response.HasField("task"):
            final_task = response.task

    # 5. The deliverable lives in the first artifact's parts.
    return get_artifact_text(final_task.artifacts[0])


async def main() -> None:
    """Run the two-hop chain: topic -> Research -> Writer -> printed result."""
    async with httpx.AsyncClient() as http_client:
        print(f"Topic: {TOPIC}\n")

        print("Step 1 - hiring the Research Agent")
        summary = await call_agent(http_client, RESEARCH_URL, TOPIC)
        print(f"\nResearch summary:\n{summary}\n")

        print("Step 2 - hiring the Writer Agent")
        rewrite = await call_agent(http_client, WRITER_URL, summary)
        print(f"\nFinal punchy explanation:\n{rewrite}\n")


if __name__ == "__main__":
    asyncio.run(main())
