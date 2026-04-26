# Architecture — Voice Booking Agent

## Call Flow
```
Caller (Phone)
    ↓
Twilio SIP Trunk
    ↓
LiveKit SIP (handles telephony session)
    ↓
LiveKit Room (audio streaming)
    ↓
Agent Worker (Python process)
    ├── Sarvam STT (saarika — Indian languages) → transcript
    ├── LiveKit VAD (built-in turn detection)
    ├── LangGraph Agent (intent + conversation logic)
    │       ↓
    │   Your Booking Backend (FastAPI — pluggable)
    └── Sarvam TTS (bulbul — Indian languages) → audio back to caller
```

## Design Principles
- Generic first: no business logic hardcoded — all config driven
- Pluggable backend: swap the booking API without touching voice layer
- Language flexible: switch between English and Indian languages via config
- Use LiveKit native Sarvam plugin — do NOT call Sarvam APIs directly

## Key Design Decisions
- AgentSession is the orchestrator (VoicePipelineAgent is deprecated — always verify)
- LangGraph handles conversation state — LiveKit handles voice only
- One LangGraph graph instance per call session
- Booking backend is a separate FastAPI service (can be swapped for any business)

## Audio Format Chain
LiveKit Room (Opus/WebRTC) → Agent Worker → PCM 16kHz → Sarvam STT
Sarvam TTS → PCM/WAV → Agent Worker → LiveKit → Caller

## Latency Budget (target under 1.5s total)
- VAD end detection: ~100ms
- Sarvam STT: ~200-400ms
- LangGraph + booking API: ~400-800ms
- Sarvam TTS first chunk: ~200-300ms
- Stream TTS sentence by sentence — never wait for full response

## Booking Backend (pluggable)
Define a standard interface. Any business plugs in their own implementation:
- GET  /slots?date_from=...&date_to=...&provider_id=...
- POST /bookings (create)
- PUT  /bookings/{id} (reschedule)
- DELETE /bookings/{id} (cancel)
- GET  /bookings?customer_name=...

## Environment Variables
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
SARVAM_API_KEY=...
BOOKING_API_BASE=http://localhost:8000
DEFAULT_LANGUAGE=en-IN
BUSINESS_NAME=My Business

## Project Structure
voice-booking-agent/
├── CLAUDE.md              ← Claude Code instructions (session brain)
├── PHASES.md              ← Phase plan with showable outcomes
├── ARCHITECTURE.md        ← This file
├── PHASE_1.md             ← Phase 1 detailed steps
├── PHASE_2.md             ← Phase 2 detailed steps
├── PHASE_3.md             ← Phase 3 detailed steps
├── PHASE_4.md             ← Phase 4 detailed steps
├── PHASE_5.md             ← Phase 5 detailed steps
├── .env.example           ← Env vars template
├── requirements.txt        ← Pinned deps (updated per phase after version check)
└── src/
    ├── agent.py            ← Main agent entry point
    ├── config.py           ← Config and env loading
    ├── conversation/
    │   ├── graph.py        ← LangGraph conversation graph
    │   ├── nodes.py        ← LangGraph nodes (intents)
    │   └── tools.py        ← Booking API tool calls
    └── booking/
        ├── client.py       ← Generic booking API client
        └── mock.py         ← Mock booking backend for dev/testing

