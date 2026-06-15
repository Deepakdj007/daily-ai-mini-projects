"""Central configuration for the real-time voice translator.

Loads the Gemini API key from .env and defines audio rate constants,
the translate model ID, and the full language menu shown in the UI.
"""

import os

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# Dedicated Gemini translate model — not a regular assistant model.
# Only TranslationConfig works here; system_instruction is not supported.
GEMINI_MODEL: str = "gemini-3.5-live-translate-preview"

# Audio rate constants.
# Gemini Live expects 16 kHz PCM input and returns 24 kHz PCM output.
# The browser's WebRTC stack always runs at 48 kHz — resampling is mandatory.
INPUT_RATE: int = 16_000   # mic → Gemini
OUTPUT_RATE: int = 24_000  # Gemini → speaker (before resampling)
WEBRTC_RATE: int = 48_000  # browser WebRTC standard

# Languages shown in the target-language selector.
# Keys are display names; values are BCP-47 codes the Gemini API accepts.
LANGUAGES: dict[str, str] = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Malayalam": "ml",
    "Kannada": "kn",
    "Bengali": "bn",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Punjabi": "pa",
    "Urdu": "ur",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese (Simplified)": "zh",
    "Arabic": "ar",
    "Portuguese": "pt",
    "Russian": "ru",
}
