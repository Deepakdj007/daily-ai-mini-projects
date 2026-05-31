# voice-ai-agent (Vani)

Real-time voice assistant on LiveKit Agents with swappable STT / LLM / TTS presets.

## Quick start

```bash
cp .env.example .env
# Fill in API keys, then:
uv run python src/agent.py download-files
uv run python src/agent.py dev
```

Open **LiveKit Cloud → Agents → Agent Console** and start a session.

## Pipeline presets

Set one preset in `.env`:

```env
VOICE_PIPELINE_PRESET=sarvam_gemini
```

| Preset | STT | LLM | TTS | Language | Keys |
|--------|-----|-----|-----|----------|------|
| `current_en` | Deepgram Nova-3 | Gemini Flash Lite | Deepgram Aura | English | LIVEKIT, DEEPGRAM, GOOGLE |
| `sarvam_gemini` | Sarvam Saaras v3 (codemix) | Gemini | Sarvam Bulbul | Manglish | + SARVAM |
| `sarvam_googletts` | Sarvam Saaras v3 (codemix) | Gemini | Google Gemini 3.1 Flash TTS Preview | Manglish | + SARVAM, GCP creds |
| `soniox_gemini` | Soniox (ml+en hints) | Gemini | Soniox | Manglish | + SONIOX |
| `sarvam_full` | Sarvam | Sarvam 30B | Sarvam Bulbul | Manglish | SARVAM, LIVEKIT |
| `soniox_full` | Soniox | Gemini | Soniox | Manglish | SONIOX, GOOGLE |
| `gcp_gemini` | Google Chirp 3 | Gemini | Google Gemini TTS | Manglish | GCP creds, GOOGLE |
| `hybrid_budget` | Sarvam | Gemini | Soniox | Manglish | SARVAM, SONIOX, GOOGLE |

### Mix-and-match overrides

```env
VOICE_PIPELINE_PRESET=sarvam_gemini
STT_PROVIDER=sarvam
LLM_PROVIDER=google
TTS_PROVIDER=soniox
LANGUAGE_PROFILE=manglish
```

`LANGUAGE_PROFILE`: `en` | `ml` | `manglish`

## Manglish QA checklist

Manual tests in Agent Console (use headphones):

- [ ] Pure Malayalam question → reply in Malayalam script
- [ ] Pure English question → reply in English
- [ ] Manglish sentence → reply matches user's mix
- [ ] Say "stop" while agent speaks → brief acknowledgment
- [ ] Noisy background → BVC improves STT (compare off if needed)

Log line on connect shows active preset:

```text
pipeline preset=sarvam_gemini stt=sarvam llm=google tts=sarvam language=manglish
```

## Notes

- Deepgram STT does **not** support Malayalam; use `sarvam_gemini` or `soniox_gemini` for Manglish.
- `google.TTS` requires **Google Cloud** credentials (`GOOGLE_APPLICATION_CREDENTIALS`), not AI Studio key alone.
- Do not set `detect_language=True` on Deepgram streaming STT.
