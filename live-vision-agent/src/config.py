"""Central configuration for the live vision agent.

Loads the Gemini API key from .env and defines every constant the
agent needs: model ID, audio formats, camera settings, and the
system prompt that shapes the agent's personality.
"""

import os

import pyaudio
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# Latest Live API model (free tier) — native audio in/out, vision input.
MODEL: str = "gemini-3.1-flash-live-preview"
VOICE: str = "Zephyr"

# Live API audio contract: mic input must be 16 kHz mono 16-bit PCM,
# the model speaks back at 24 kHz mono 16-bit PCM.
AUDIO_FORMAT: int = pyaudio.paInt16
CHANNELS: int = 1
SEND_SAMPLE_RATE: int = 16_000
RECEIVE_SAMPLE_RATE: int = 24_000
CHUNK_SIZE: int = 1_024

# Live API accepts at most 1 video frame per second.
FRAME_INTERVAL_SECONDS: float = 1.0
MAX_FRAME_SIZE: tuple[int, int] = (768, 768)
JPEG_QUALITY: int = 85

SYSTEM_PROMPT: str = (
    "You are a sharp, friendly assistant with live access to the user's "
    "camera. You can see what they show you in real time. Answer out loud, "
    "keep responses short and conversational, and when the user shows you "
    "something, describe or reason about what you actually see."
)
