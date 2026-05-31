# src/agent.py
# ─────────────────────────────────────────────────────────────
# Real-time voice AI agent using LiveKit's STT-LLM-TTS pipeline.
# Pipeline providers are swappable via VOICE_PIPELINE_PRESET in .env.
# ─────────────────────────────────────────────────────────────

import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RoomInputOptions,
    cli,
    WorkerOptions,
)
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from pipeline_config import (
    build_pipeline,
    get_assistant_instructions,
    get_greeting_instruction,
)

load_dotenv()
logger = logging.getLogger("voice-ai-agent")


class VoiceAssistant(Agent):
    """Concise conversational voice assistant (Vani)."""

    def __init__(self, *, language_profile: str = "en") -> None:
        super().__init__(instructions=get_assistant_instructions(language_profile))  # type: ignore[arg-type]


async def entrypoint(ctx: JobContext) -> None:
    """Build the STT → LLM → TTS pipeline and connect to the LiveKit room."""
    await ctx.connect()

    stt, llm, tts, resolved = build_pipeline()
    logger.info(
        "pipeline preset=%s stt=%s llm=%s tts=%s language=%s",
        resolved.preset,
        resolved.stt_provider,
        resolved.llm_provider,
        resolved.tts_provider,
        resolved.language_profile,
    )

    session = AgentSession(
        stt=stt,
        llm=llm,
        tts=tts,
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=VoiceAssistant(language_profile=resolved.language_profile),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await session.generate_reply(
        instructions=get_greeting_instruction(resolved.language_profile)  # type: ignore[arg-type]
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
