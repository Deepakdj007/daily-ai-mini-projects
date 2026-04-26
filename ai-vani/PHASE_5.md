# Phase 5 — Production Hardening

**Goal:** Reliable calls, good latency, proper logging.
**Showable:** 10 back-to-back test calls, all complete cleanly, latency logged.
**Cost:** Check LiveKit Cloud pricing for production tier if needed.

---

## Checklist before starting this phase
- [ ] Phases 1-4 all acceptance criteria passing
- [ ] At least 20 test calls done manually

## What to harden

### Interruption handling
- User speaks while agent speaks → agent stops immediately
- LiveKit AgentSession has allow_interruptions param — verify latest default
- Check: https://docs.livekit.io/agents/voice-agent/

### Streaming TTS
- Don't wait for full LangGraph response before starting TTS
- Stream sentence by sentence
- Target: first audio chunk < 800ms after user stops speaking

### API failure fallbacks
For each HealthSigns API call, define fallback behaviour:
- patient-lookup fails → "Let me connect you to a colleague"
- available-slots fails → offer callback
- book-appointment fails → retry once, then human handoff

### Call logging
- Log: call start, intent detected, APIs called, call end, total duration
- Store: transcript per call (to DB or file)
- Post-call summary (mirrors Retell's post_call_analysis)

### Human handoff
- Transfer number: confirm with user what number to use
- Implement LiveKit SIP transfer (check docs before coding)

### Timeout handling
- User silent for 10s → gentle prompt
- User silent for 20s → end call politely

## Acceptance criteria
- [ ] 10 consecutive calls complete without crash
- [ ] Average latency < 1.5s (logged)
- [ ] Interruption stops agent within 300ms
- [ ] Human transfer works
- [ ] Call log written for every call

