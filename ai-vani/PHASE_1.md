# Phase 1 — LiveKit Hello World

**Goal:** Agent joins LiveKit room, speaks greeting, echoes back what user says.
**Showable:** LiveKit Playground → hear agent → speak → hear echo.
**Cost:** FREE (LiveKit Cloud dev tier + Sarvam free tier)

---

## Before writing any code — check these docs
1. https://docs.livekit.io/agents/quickstart/ (Voice AI quickstart)
2. https://docs.livekit.io/reference/python/livekit/plugins/sarvam/index.html (Sarvam plugin)
3. https://pypi.org/project/livekit-agents/ (latest version)
4. https://pypi.org/project/livekit-plugins-sarvam/ (latest version)

## Step-by-step instructions (instruct user, don't code until asked)

### Step 1 — LiveKit Cloud account
Tell user to:
- Go to https://livekit.io → sign up free
- Create a project
- Copy: URL (wss://...), API Key, API Secret
- No credit card needed for dev tier

### Step 2 — Sarvam API key
Tell user to:
- Go to https://dashboard.sarvam.ai → sign up
- Copy API key
- Free tier available

### Step 3 — Python environment
Instruct user to create a venv, then check latest versions of:
- livekit-agents (check pypi first)
- livekit-plugins-sarvam (check pypi first)
- livekit (check pypi first)
THEN write requirements.txt with pinned versions.

### Step 4 — .env file
Create .env from .env.example. Fill in LiveKit + Sarvam keys.

### Step 5 — Agent file
Write src/agent.py using:
- AgentSession (NOT VoicePipelineAgent — deprecated)
- livekit.plugins.sarvam STT and TTS
- On connect: speak a greeting in English
- On user speech: echo the transcript back

### Step 6 — Run and test
Instruct user to:
- Run: python src/agent.py dev
- Open: https://agents-playground.livekit.io
- Connect to their project
- Speak and verify echo

## Acceptance criteria (must all pass before Phase 2)
- [ ] Agent connects to LiveKit room without error
- [ ] Agent speaks greeting on connect
- [ ] User speech is transcribed (check logs)
- [ ] Agent echoes transcript back audibly
- [ ] No crashes on 3 consecutive test calls

