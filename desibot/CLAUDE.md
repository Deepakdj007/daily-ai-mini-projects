# VoiceBot Demo — CLAUDE.md

## Project Overview
A demo voice bot app that makes outbound calls using **Retell AI** (telephony + agent orchestration) with **Sarvam AI** (Indian language STT/TTS). The app makes calls, gets transcriptions, and extracts structured data variables from conversations.

**Stack:** FastAPI (backend) + Vite + React (frontend)

---

## Architecture

```
User (React UI)
  → FastAPI Backend
      → Retell AI SDK  (outbound call trigger, webhook events)
      → Sarvam AI API  (STT, TTS, optional translate)
      → In-memory / SQLite store (call records, extracted data)
```

### Call Flow
1. User enters phone number + call goal in React UI
2. FastAPI calls Retell `create_phone_call` API
3. Retell connects the call using your registered agent
4. During call → Retell sends real-time transcript via webhook
5. On call end → FastAPI extracts variables using LLM (Sarvam 30B — free)
6. React UI polls for call status and shows transcript + extracted data

---

## Folder Structure

```
voicebot-demo/
├── backend/
│   ├── main.py                  # FastAPI app entry
│   ├── routers/
│   │   ├── calls.py             # POST /calls/initiate, GET /calls/{id}
│   │   └── webhooks.py          # POST /webhooks/retell
│   ├── services/
│   │   ├── retell_service.py    # Retell SDK wrapper
│   │   ├── sarvam_service.py    # Sarvam STT/TTS/LLM wrapper
│   │   └── extractor.py        # Variable extraction from transcript
│   ├── models/
│   │   └── call.py              # Pydantic models: CallRecord, ExtractedData
│   ├── db.py                    # Simple SQLite via aiosqlite
│   └── config.py                # Settings via pydantic-settings
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── pages/
│   │   │   ├── Dashboard.tsx    # List all calls
│   │   │   └── CallDetail.tsx   # Transcript + extracted vars
│   │   ├── components/
│   │   │   ├── InitiateCallForm.tsx
│   │   │   ├── CallCard.tsx
│   │   │   └── TranscriptViewer.tsx
│   │   └── api/
│   │       └── client.ts        # Axios API client
│   ├── index.html
│   └── vite.config.ts
├── CLAUDE.md                    # ← you are here
├── CONVENTIONS.md
├── COSTS.md
└── .env.example
```

---

## Key Commands

### Backend
```bash
cd backend
uv sync
uv run uvicorn main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev          # Vite dev server on :5173
```

### Expose localhost for Retell webhooks (dev)
```bash
ngrok http 8000
# Then set RETELL_WEBHOOK_URL=https://<ngrok-id>.ngrok.io/webhooks/retell
```

---

## Environment Variables

See `.env.example`. Required keys:
```
RETELL_API_KEY=          # From retell.ai dashboard
SARVAM_API_KEY=          # From dashboard.sarvam.ai
RETELL_AGENT_ID=         # Create an agent in Retell dashboard first
RETELL_PHONE_NUMBER=     # Buy/register a number in Retell dashboard
RETELL_WEBHOOK_URL=      # Your ngrok or deployed URL
```

---

## Retell Setup (Do This First)

Before running the app, in Retell dashboard:
1. Create an Agent → choose "Custom LLM" → point it to your FastAPI LLM endpoint (`/retell/llm`)
2. Buy or import a phone number
3. Register webhook URL pointing to `/webhooks/retell`
4. Note the `agent_id` and phone number

---

## Sarvam AI Usage in This Project

| Feature | Sarvam API | Cost |
|---|---|---|
| LLM (variable extraction) | Sarvam 30B | **Free** |
| TTS (agent voice) | Bulbul v2 | ₹15/10K chars |
| STT (transcription fallback) | Speech to Text | ₹30/hour |

> **Note:** Retell handles STT/TTS internally using its own engines. Sarvam STT/TTS is used optionally for Indian language support or as a fallback. Sarvam 30B (free) is used for post-call data extraction.

---

## Data Extraction

After each call, `extractor.py` sends the full transcript to Sarvam 30B with a structured prompt to extract variables defined in `EXTRACTION_SCHEMA.md`.

Default extracted fields:
- `patient_name`
- `appointment_date`
- `chief_complaint`
- `confirmed` (boolean)
- `callback_requested` (boolean)

These are configurable per call type via the UI.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/calls/initiate` | Start an outbound call |
| GET | `/calls/` | List all calls |
| GET | `/calls/{call_id}` | Get call detail + transcript + extracted data |
| POST | `/webhooks/retell` | Retell event webhook receiver |
| POST | `/retell/llm` | Custom LLM endpoint (Retell calls this during the call) |

---

## Coding Conventions

See `CONVENTIONS.md` for full details. Quick rules:
- All backend async (FastAPI async routes, aiosqlite)
- Pydantic v2 models for all request/response shapes
- No hardcoded strings — use `config.py` / env vars
- Frontend: React functional components + TypeScript, Tailwind CSS
- No class components in React
- API errors → return proper HTTP status + `{"detail": "..."}` JSON

---

## Cost Reality Check

See `COSTS.md` for full breakdown. TL;DR for demo:
- **Retell free tier:** $10 credits ≈ ~65–90 call minutes
- **Sarvam free tier:** 999 credits (LLM is free, STT/TTS is paid)
- **Total demo budget:** ~90 minutes of calls at zero cash spend

---

## Common Gotchas

1. **Retell webhooks need a public URL** — use ngrok locally
2. **Retell Custom LLM endpoint must respond within 1 second** — keep `/retell/llm` fast
3. **Sarvam 30B is free but rate-limited** — add retry with backoff in `sarvam_service.py`
4. **Phone numbers in India** — Retell's Twilio numbers may not call Indian mobiles cheaply; test with your own SIP or Twilio number
5. **Transcript arrives via webhook**, not in the `create_phone_call` response — always use webhooks for real-time data
