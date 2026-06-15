"""Central configuration for the AI interview coach.

Loads the Gemini API key from .env and defines the model ID, audio
constants, VAD tuning, question bank, and the system prompt that
turns the Live API into an interview coach.
"""

import os

import pyaudio
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

MODEL: str = "gemini-3.1-flash-live-preview"
VOICE: str = "Charon"

# The Live model is WebSocket-only (bidiGenerateContent). The post-session
# report is a plain REST call, so it uses a standard text model instead.
REPORT_MODEL: str = "gemini-2.5-flash"

# Live API audio contract: mic 16 kHz mono 16-bit in, model 24 kHz out.
AUDIO_FORMAT: int = pyaudio.paInt16
CHANNELS: int = 1
SEND_SAMPLE_RATE: int = 16_000
RECEIVE_SAMPLE_RATE: int = 24_000
CHUNK_SIZE: int = 1_024

# Camera: Live API accepts at most 1 frame per second.
FRAME_INTERVAL_SECONDS: float = 1.0
MAX_FRAME_SIZE: tuple[int, int] = (768, 768)
JPEG_QUALITY: int = 85

# VAD: give the candidate 3 s of silence before the coach responds.
# Low end-of-speech sensitivity prevents cutting in on thinking pauses.
SILENCE_DURATION_MS: int = 3_000
PREFIX_PADDING_MS: int = 300

INTERVIEW_QUESTIONS: list[str] = [
    "Tell me about yourself and what makes you a strong candidate for this role.",
    "Describe a time you faced a major challenge at work and how you resolved it.",
    "Where do you see yourself in five years?",
    "Tell me about a project you led and the outcome.",
    "How do you handle disagreement with a teammate or manager?",
    "Describe a situation where you had to learn something quickly under pressure.",
    "What is your greatest professional achievement so far?",
    "How do you prioritise when you have multiple competing deadlines?",
    "Tell me about a failure and what you learnt from it.",
    "Why do you want to leave your current role?",
]

_QUESTION_LIST = "\n".join(
    f"{i+1}. {q}" for i, q in enumerate(INTERVIEW_QUESTIONS)
)

SYSTEM_PROMPT: str = f"""You are an expert interview coach running a live mock interview session.

You have access to the candidate's webcam (watch their body language, eye contact, posture, gestures) and their microphone (listen to their voice, tone, pace, filler words).

Your question bank for this session:
{_QUESTION_LIST}

Session flow:
1. Greet the candidate warmly and briefly explain what you will be coaching them on.
2. Ask question 1. Wait for the candidate to finish their full answer.
3. After they finish, deliver focused coaching (under 45 seconds) covering:
   - CONTENT: Was the answer complete? Did they use the STAR method for behavioural questions?
   - DELIVERY: Pace (too fast/slow?), volume, energy, confidence in their voice.
   - FILLER WORDS: Call out any "um", "uh", "like", "you know", "basically", "actually" you heard — give the count.
   - BODY LANGUAGE: What you see on camera — eye contact, posture, hand gestures, facial expression.
   - ONE IMPROVEMENT: End with a single, concrete action they can apply to the very next answer.
4. Immediately ask the next question to keep momentum.
5. After all questions, give a brief closing summary of their strongest quality and their single biggest area to work on.

Coaching tone: direct, warm, and specific. No vague praise. No academic language. Real coaches name the exact problem and the exact fix.

While the candidate is speaking, stay completely silent and observe. Only respond after they have finished their answer."""

# Sent once after connecting to make the coach speak first — greet the
# candidate and ask the opening question without waiting for input.
KICKOFF_MESSAGE: str = (
    "I'm ready to start the mock interview. Greet me warmly, briefly explain "
    "what you'll coach me on, then ask me the first question."
)
