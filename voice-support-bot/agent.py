from __future__ import annotations

import logging
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import (
    AgentSession,
    Agent,
    AgentServer,
    RoomInputOptions,
    function_tool,
)
from livekit.plugins import deepgram, silero
from livekit.plugins import groq
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins import noise_cancellation

load_dotenv(".env.local")
logger = logging.getLogger("voice-support-bot")


class SupportAgent(Agent):
    """AI voice agent for customer support."""

    def __init__(self):
        super().__init__(
            instructions=(
                "You are a friendly customer support agent for DataScienceBrain. "
                "Help users with subscriptions, content access, and billing. "
                "Keep every response to TWO sentences maximum. "
                "You are speaking out loud — never use markdown, bullet points, or lists."
            )
        )

    @function_tool
    async def check_subscription_status(self, user_email: str) -> str:
        """
        Check if a user has an active subscription.
        Call this when the user asks about their account or subscription.
        """
        return f"Account {user_email} has an active subscription."

    @function_tool
    async def create_support_ticket(self, user_email: str, issue_summary: str) -> str:
        """
        Create a support ticket. Call when the issue needs human follow-up
        or the user requests to speak with a human agent.
        """
        ticket_id = "TKT-" + str(abs(hash(user_email + issue_summary)))[-6:]
        return (
            f"Ticket {ticket_id} created. "
            f"Our team will email {user_email} within 24 hours."
        )


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(model="nova-3"),
        llm=groq.LLM(model="llama-3.3-70b-versatile"),
        tts=deepgram.TTS(model="aura-2-thalia-en"),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=SupportAgent(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await session.generate_reply(
        instructions="Greet the caller warmly and ask how you can help today."
    )

    @ctx.room.on("participant_disconnected")
    def on_disconnect(participant):
        logger.info(f"Call ended: {participant.identity}")


if __name__ == "__main__":
    agents.cli.run_app(server)
