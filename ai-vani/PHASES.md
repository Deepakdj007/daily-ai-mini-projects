# Build Phases — Voice Booking Agent

Each phase must produce a showable, working outcome before moving to the next.

---

## Phase 1 — LiveKit Hello World
**Goal:** Agent joins a LiveKit room, speaks a greeting, listens, echoes back.
**Showable:** Open LiveKit Playground → hear agent speak → speak back → hear echo.
**Cost:** FREE — LiveKit Cloud dev tier + Sarvam free tier. No Twilio needed yet.
**Builds:** Python env, LiveKit + Sarvam wired up, AgentSession with echo logic.

---

## Phase 2 — Phone Call In
**Goal:** Real phone call reaches the agent.
**Showable:** Call a phone number → agent picks up → greets you.
**Cost:** Twilio free trial ($15 credit). CONFIRM WITH USER before any setup.
**Builds:** Twilio SIP trunk → LiveKit SIP → agent dispatch rule.

---

## Phase 3 — Indian Language Support
**Goal:** Caller speaks Malayalam (or Hindi) → agent understands + replies in same language.
**Showable:** Speak Malayalam into phone → agent replies in Malayalam.
**Cost:** Sarvam free tier. Check current limits at dashboard.sarvam.ai/pricing.
**Builds:** Language config (ml-IN), Sarvam STT + TTS in Malayalam, transcript logging.

---

## Phase 4 — Real Booking Conversation
**Goal:** Full conversation: caller books, cancels, or reschedules an appointment.
**Showable:** Complete call from greeting → intent → slot selection → confirmed booking.
**Cost:** FREE — uses mock booking backend in dev, real backend when ready.
**Builds:** LangGraph graph, booking tool calls, mock backend, session state per call.

---

## Phase 5 — Production Hardening
**Goal:** Reliable, low-latency calls with proper fallbacks and logging.
**Showable:** 10 back-to-back test calls, all complete cleanly, latency under 1.5s.
**Cost:** Check LiveKit Cloud production pricing if scaling beyond dev tier.
**Builds:** Interruption handling, streaming TTS, error fallbacks, call logging, human transfer.

