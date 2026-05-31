"""Verify pipeline preset resolution (no API calls)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline_config import (  # noqa: E402
    PRESET_DEFAULTS,
    get_assistant_instructions,
    get_greeting_instruction,
    resolve_pipeline,
)


def main() -> None:
    for preset_name, (stt, llm, tts, lang) in PRESET_DEFAULTS.items():
        os.environ["VOICE_PIPELINE_PRESET"] = preset_name
        os.environ.pop("STT_PROVIDER", None)
        os.environ.pop("LLM_PROVIDER", None)
        os.environ.pop("TTS_PROVIDER", None)
        os.environ.pop("LANGUAGE_PROFILE", None)

        resolved = resolve_pipeline()
        assert resolved.preset == preset_name, (preset_name, resolved.preset)
        assert resolved.stt_provider == stt
        assert resolved.llm_provider == llm
        assert resolved.tts_provider == tts
        assert resolved.language_profile == lang

        get_assistant_instructions(resolved.language_profile)
        get_greeting_instruction(resolved.language_profile)
        print(f"OK  {preset_name} -> stt={stt} llm={llm} tts={tts} lang={lang}")

    print(f"\nAll {len(PRESET_DEFAULTS)} presets verified.")


if __name__ == "__main__":
    main()
