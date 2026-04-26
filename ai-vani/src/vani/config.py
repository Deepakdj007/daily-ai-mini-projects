import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# LiveKit
LIVEKIT_URL = os.environ["LIVEKIT_URL"]
LIVEKIT_API_KEY = os.environ["LIVEKIT_API_KEY"]
LIVEKIT_API_SECRET = os.environ["LIVEKIT_API_SECRET"]

# Sarvam AI — STT, TTS, LLM all use the same key
SARVAM_API_KEY = os.environ["SARVAM_API_KEY"]
SARVAM_LLM_BASE_URL = os.getenv("SARVAM_LLM_BASE_URL", "https://api.sarvam.ai/v1")
SARVAM_LLM_MODEL = os.getenv("SARVAM_LLM_MODEL", "sarvam-m")  # override in .env to sarvam-105b for tool calling

# Agent
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "en-IN")
BUSINESS_NAME = os.getenv("BUSINESS_NAME", "My Booking Agent")

# Database
DB_PATH = Path(os.getenv("DB_PATH", "data/vani.db"))
