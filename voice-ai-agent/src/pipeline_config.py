"""Swappable STT / LLM / TTS presets for voice-ai-agent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

from livekit.plugins import deepgram, google, sarvam, soniox

LanguageProfile = Literal["en", "ml", "manglish"]
SttProvider = Literal["deepgram", "sarvam", "soniox", "google"]
LlmProvider = Literal["google", "sarvam"]
TtsProvider = Literal["deepgram", "sarvam", "soniox", "google"]

PRESET_DEFAULTS: dict[str, tuple[SttProvider, LlmProvider, TtsProvider, LanguageProfile]] = {
    "current_en": ("deepgram", "google", "deepgram", "en"),
    "sarvam_gemini": ("sarvam", "google", "sarvam", "manglish"),
    "sarvam_googletts": ("sarvam", "google", "google", "manglish"),
    "soniox_gemini": ("soniox", "google", "soniox", "manglish"),
    "sarvam_full": ("sarvam", "sarvam", "sarvam", "manglish"),
    "soniox_full": ("soniox", "google", "soniox", "manglish"),
    "gcp_gemini": ("google", "google", "google", "manglish"),
    "hybrid_budget": ("sarvam", "google", "soniox", "manglish"),
}

DEFAULT_PRESET = "current_en"
DEFAULT_LANGUAGE_PROFILE = "en"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "80"))
SARVAM_LLM_MODEL = os.getenv("SARVAM_LLM_MODEL", "sarvam-30b")
SONIOX_TTS_VOICE = os.getenv("SONIOX_TTS_VOICE", "Maya")
# Default to the newest Gemini TTS preview for gcp_gemini; can be overridden in .env
GOOGLE_TTS_MODEL = os.getenv("GOOGLE_TTS_MODEL", "gemini-3.1-flash-tts-preview")
GOOGLE_TTS_VOICE_NAME = os.getenv("GOOGLE_TTS_VOICE_NAME", "Achernar")
GOOGLE_STT_LOCATION = os.getenv("GOOGLE_STT_LOCATION", "asia-southeast1")


@dataclass(frozen=True)
class ResolvedPipeline:
    preset: str
    stt_provider: SttProvider
    llm_provider: LlmProvider
    tts_provider: TtsProvider
    language_profile: LanguageProfile


def resolve_pipeline() -> ResolvedPipeline:
    requested = os.getenv("VOICE_PIPELINE_PRESET", DEFAULT_PRESET).strip().lower()
    preset = requested

    if preset in PRESET_DEFAULTS:
        stt_p, llm_p, tts_p, lang = PRESET_DEFAULTS[preset]
    else:
        import logging

        logging.getLogger("voice-ai-agent").warning(
            "Unknown VOICE_PIPELINE_PRESET=%r, using %r", requested, DEFAULT_PRESET
        )
        preset = DEFAULT_PRESET
        stt_p, llm_p, tts_p, lang = PRESET_DEFAULTS[DEFAULT_PRESET]

    stt_provider = os.getenv("STT_PROVIDER", stt_p).strip().lower()  # type: ignore[assignment]
    llm_provider = os.getenv("LLM_PROVIDER", llm_p).strip().lower()  # type: ignore[assignment]
    tts_provider = os.getenv("TTS_PROVIDER", tts_p).strip().lower()  # type: ignore[assignment]
    language_profile = os.getenv("LANGUAGE_PROFILE", lang).strip().lower()  # type: ignore[assignment]

    return ResolvedPipeline(
        preset=preset,
        stt_provider=stt_provider,  # type: ignore[arg-type]
        llm_provider=llm_provider,  # type: ignore[arg-type]
        tts_provider=tts_provider,  # type: ignore[arg-type]
        language_profile=language_profile,  # type: ignore[arg-type]
    )


def build_stt(resolved: ResolvedPipeline) -> Any:
    profile = resolved.language_profile

    if resolved.stt_provider == "deepgram":
        language = "en-IN" if profile in ("en", "manglish") else "en-IN"
        return deepgram.STT(model="nova-3", language=language)

    if resolved.stt_provider == "sarvam":
        if profile == "en":
            return sarvam.STT(language="en-IN", model="saaras:v3", mode="transcribe")
        return sarvam.STT(language="ml-IN", model="saaras:v3", mode="codemix")

    if resolved.stt_provider == "soniox":
        hints = ["ml", "en"] if profile in ("ml", "manglish") else ["en"]
        return soniox.STT(
            params=soniox.STTOptions(
                language_hints=hints,
                enable_language_identification=True,
            )
        )

    if resolved.stt_provider == "google":
        languages = ["ml-IN"] if profile in ("ml", "manglish") else ["en-IN"]
        return google.STT(
            languages=languages,
            model="chirp_3",
            location=GOOGLE_STT_LOCATION,
            detect_language=False,
        )

    raise ValueError(f"Unknown STT provider: {resolved.stt_provider}")


def build_llm(resolved: ResolvedPipeline) -> Any:
    if resolved.llm_provider == "sarvam":
        return sarvam.LLM(model=SARVAM_LLM_MODEL, max_tokens=GEMINI_MAX_OUTPUT_TOKENS)

    return google.LLM(model=GEMINI_MODEL, max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS)


def build_tts(resolved: ResolvedPipeline) -> Any:
    profile = resolved.language_profile

    if resolved.tts_provider == "deepgram":
        return deepgram.TTS(model="aura-2-thalia-en")

    if resolved.tts_provider == "sarvam":
        lang = "ml-IN" if profile in ("ml", "manglish") else "en-IN"
        return sarvam.TTS(target_language_code=lang, model="bulbul:v3")

    if resolved.tts_provider == "soniox":
        lang = "ml" if profile in ("ml", "manglish") else "en"
        return soniox.TTS(language=lang, voice=SONIOX_TTS_VOICE)

    if resolved.tts_provider == "google":
        lang = "ml-IN" if profile in ("ml", "manglish") else "en-US"
        return google.TTS(
            language=lang,
            model_name=GOOGLE_TTS_MODEL,
            voice_name=GOOGLE_TTS_VOICE_NAME,
            gender="female",
        )

    raise ValueError(f"Unknown TTS provider: {resolved.tts_provider}")


def build_pipeline() -> tuple[Any, Any, Any, ResolvedPipeline]:
    resolved = resolve_pipeline()
    return build_stt(resolved), build_llm(resolved), build_tts(resolved), resolved


def get_assistant_instructions(language_profile: LanguageProfile) -> str:
    base = (
        "You are Vani, a friendly voice assistant on a live phone-style call. "
        "Keep every reply to one or two short spoken sentences. "
        "Never mention being an AI, LLM, or who trained you. "
        "If asked your name, say Vani. "
        "If asked capabilities, give one brief example, not a feature list. "
        "Never use bullet points, markdown, or numbered lists."
    )

    if language_profile == "en":
        return (
            base
            + " If the user says stop, wait, or hold on, reply with only: Sure, go ahead."
        )

    if language_profile == "ml":
        return (
            base
            + " Always reply in Malayalam using Malayalam script. "
            "If the user says stop, wait, or hold on, reply with only: ശരി, പറയൂ."
        )

    return (
        base
        + " Reply in the same language mix the user uses (English, Malayalam, or Manglish). "
        "Use Malayalam script for Malayalam words; do not romanize unless the user did. "
        "If the user says stop, wait, or hold on, reply briefly: Sure, go ahead. or ശരി, പറയൂ."
    )


def get_greeting_instruction(language_profile: LanguageProfile) -> str:
    if language_profile == "ml":
        return (
            "Greet in Malayalam: say you are Vani and ask what they need — one short sentence only."
        )
    if language_profile == "manglish":
        return (
            "Greet in friendly Manglish (short): say hi, you're Vani, ask what they need — "
            "one sentence only."
        )
    return "Say hi, you're Vani, and ask what they need — one short sentence only."
