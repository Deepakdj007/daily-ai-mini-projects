# TASKS.md — Implementation Checklist

Claude Code should work through these in order. Check off each task when done.

---

## Phase 1 — Backend Skeleton

- [ ] `backend/config.py` — pydantic-settings with all env vars
- [ ] `backend/db.py` — aiosqlite setup, create `calls` table on startup
- [ ] `backend/models/call.py` — `CallRecord`, `CallStatus` enum, `ExtractedData` Pydantic models
- [ ] `backend/main.py` — FastAPI app, CORS, lifespan, include routers

## Phase 2 — Retell Integration

- [ ] `backend/services/retell_service.py` — wrap Retell Python SDK
  - `create_phone_call(to_number, from_number, agent_id, metadata)`
  - `get_call(call_id)`
- [ ] `backend/routers/calls.py`
  - `POST /calls/initiate` — validates input, calls retell_service, saves to DB
  - `GET /calls/` — list all calls from DB
  - `GET /calls/{call_id}` — get single call with transcript + extracted data
- [ ] `backend/routers/webhooks.py`
  - `POST /webhooks/retell` — handle `call_started`, `call_ended`, `transcript` events
  - On `call_ended` → trigger extraction pipeline

## Phase 3 — Sarvam AI Integration

- [ ] `backend/services/sarvam_service.py`
  - `chat_completion(messages)` — call Sarvam 30B (free)
  - `text_to_speech(text, language)` — Bulbul v2 (optional, for TTS)
- [ ] `backend/services/extractor.py`
  - `extract_variables(transcript, schema_type)` → returns `ExtractedData`
  - Uses Sarvam 30B with prompt from `EXTRACTION_SCHEMA.md`

## Phase 4 — Retell Custom LLM Endpoint

- [ ] `backend/routers/retell_llm.py`
  - `POST /retell/llm` — receives Retell LLM request format
  - Calls Sarvam 30B (free) to generate agent response
  - Returns Retell-compatible response format
  - This eliminates LLM costs on Retell side

## Phase 5 — Frontend

- [ ] Vite + React + TypeScript project init in `frontend/`
- [ ] `src/api/client.ts` — axios instance with base URL from env
- [ ] `src/types/index.ts` — TypeScript interfaces matching backend Pydantic models
- [ ] `src/pages/Dashboard.tsx` — list calls, initiate call form
- [ ] `src/components/InitiateCallForm.tsx` — phone number input + schema selector
- [ ] `src/pages/CallDetail.tsx` — transcript viewer + extracted data display
- [ ] `src/components/TranscriptViewer.tsx` — shows speaker-labeled transcript turns
- [ ] Polling: Dashboard auto-refreshes call list every 5s

## Phase 6 — Polish

- [ ] Add loading states and error toasts in React
- [ ] Add call status badge (initiated / in-progress / completed / failed)
- [ ] Webhook signature verification in `webhooks.py`
- [ ] README.md with setup steps
- [ ] Docker Compose (optional, for deployment)

---

## Notes for Claude Code

- Always read `CLAUDE.md` and `CONVENTIONS.md` before writing any file
- Use the folder structure from `CLAUDE.md` exactly
- Run `uv sync && uv run uvicorn main:app --reload` after Phase 1 to verify startup
- Test webhook locally with ngrok before moving to Phase 5
