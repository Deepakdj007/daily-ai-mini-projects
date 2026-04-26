"""
Entry point for the Vani booking agent.

Stack:
  STT  — Sarvam Saarika (Malayalam-optimised)
  LLM  — Sarvam-M via OpenAI-compatible plugin
  TTS  — Sarvam Bulbul (Malayalam-optimised)

Run:
  uv run src/vani/agent.py dev
"""

import logging
from pathlib import Path

from livekit.agents import Agent, AgentSession, JobContext, RoomInputOptions, WorkerOptions, cli
from livekit.plugins import openai, sarvam

from vani import db
from vani.config import DEFAULT_LANGUAGE, SARVAM_API_KEY, SARVAM_LLM_BASE_URL, SARVAM_LLM_MODEL
from vani.prompts import build_system_prompt
from vani.tools import book_appointment, check_patient, get_slots, register_patient

_LOG_FILE = Path("logs/agent.log")


def _setup_logging() -> None:
    """Configure root logger: console + file (overwrite on each run)."""
    _LOG_FILE.parent.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s  %(message)s")

    file_handler = logging.FileHandler(_LOG_FILE, mode="w", encoding="utf-8")
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(file_handler)
    root.addHandler(console_handler)


_setup_logging()
logger = logging.getLogger("vani.agent")


class BookingAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=build_system_prompt(),
            tools=[check_patient, register_patient, get_slots, book_appointment],
        )


async def entrypoint(ctx: JobContext) -> None:
    logger.info("Agent connecting to room: %s", ctx.room.name)

    await db.init_db()

    session = AgentSession(
        stt=sarvam.STT(
            language=DEFAULT_LANGUAGE,
            api_key=SARVAM_API_KEY,
        ),
        llm=openai.LLM(
            model=SARVAM_LLM_MODEL,
            base_url=SARVAM_LLM_BASE_URL,
            api_key=SARVAM_API_KEY,
        ),
        tts=sarvam.TTS(
            target_language_code=DEFAULT_LANGUAGE,
            api_key=SARVAM_API_KEY,
        ),
    )

    await session.start(
        room=ctx.room,
        agent=BookingAgent(),
        room_input_options=RoomInputOptions(),
    )

    await session.generate_reply()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
