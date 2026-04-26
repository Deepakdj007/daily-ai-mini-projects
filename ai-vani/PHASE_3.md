# Phase 3 — Malayalam STT/TTS

**Goal:** Caller speaks Malayalam → agent understands + replies in Malayalam.
**Showable:** Speak Malayalam → agent replies in Malayalam.
**Cost:** Sarvam free tier (check current limits at dashboard.sarvam.ai/pricing)

---

## Before writing any code — check these docs
1. https://docs.livekit.io/reference/python/livekit/plugins/sarvam/index.html
2. https://docs.sarvam.ai/api-reference-docs/endpoints/speech-to-text
3. https://docs.sarvam.ai/api-reference-docs/changelog (check for latest model versions)

## Key facts to verify before coding
- Malayalam language code: ml-IN
- STT model as of April 2026: saarika:v2.5
- TTS model as of April 2026: bulbul:v2 (bulbul:v1 deprecated April 30 2025)
- ALWAYS re-check these — models update frequently

## Step-by-step instructions

### Step 1 — Update STT config
Change language to ml-IN in Sarvam STT plugin.

### Step 2 — Update TTS config
Change language to ml-IN, pick a Malayalam voice from bulbul:v2.
Check available voices at dashboard.sarvam.ai before hardcoding.

### Step 3 — Test with Malayalam phrases
Test phrases:
- "എനിക്ക് ഒരു അപ്പോയിൻ്റ്മെൻ്റ് വേണം" (I want an appointment)
- "എന്റെ പേര് [name]" (My name is [name])

### Step 4 — Language detection (optional enhancement)
If caller may speak English OR Malayalam, add auto-detection using saaras:v2.5
(auto-detects input language). Discuss with user before implementing.

## Acceptance criteria
- [ ] Malayalam speech is transcribed correctly (check logs)
- [ ] Agent responds in Malayalam audio
- [ ] No degradation to English-only callers

