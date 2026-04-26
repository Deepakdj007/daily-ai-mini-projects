# COSTS.md — API Cost Reference

Track and understand costs before making API changes.

---

## Retell AI

| Component | Cost | Notes |
|---|---|---|
| Free credits | **$10** on signup | ≈ 65–90 minutes of calls |
| Voice engine (ElevenLabs/Cartesia) | $0.07/min | Default |
| LLM (Gemini 2.0 Flash) | $0.006/min | **Cheapest LLM option** |
| LLM (Custom LLM) | $0.00/min | Use your own FastAPI endpoint = no LLM cost |
| Telephony (Retell Twilio) | $0.015/min | Or use your own SIP = $0 |
| Concurrency | 20 concurrent calls free | |

### Demo Budget Estimate
Using Custom LLM (FastAPI) + Cartesia voice + Retell Twilio:
- Cost per minute ≈ $0.07 (voice) + $0.015 (telephony) = **~$0.085/min**
- $10 free credits ÷ $0.085 = **~117 minutes of demo calls**

> **Tip:** Use "Custom LLM" in Retell (your own FastAPI `/retell/llm` endpoint) so you pay $0 for LLM. Use Sarvam 30B (free) to power the LLM logic.

---

## Sarvam AI

| Service | Cost | Free Credits Usage |
|---|---|---|
| Sarvam 30B (LLM) | **Free** | No credits used |
| Sarvam 30B (LLM) | **Free** | No credits used |
| TTS Bulbul v2 | ₹15 / 10K chars | ~66K chars per 999 credits |
| STT | ₹30 / hour | ~33 hours per 999 credits |
| Translate | ₹20 / 10K chars | ~50K chars per 999 credits |

### 999 Credit Reality
- If using Sarvam only for **post-call extraction** (Sarvam 30B LLM = free), your 999 credits stay untouched
- Credits only used if you add Sarvam TTS/STT alongside Retell

---

## Total Demo Cost (Zero Cash)

| Service | Free Tier | Estimated Demo Usage |
|---|---|---|
| Retell AI | $10 free | ~100 minutes of outbound calls |
| Sarvam AI (LLM) | Free forever | Unlimited extraction calls |
| Sarvam AI (STT/TTS) | 999 credits | Use only if needed |

**You can run this entire demo for ₹0 cash investment.**

---

## Cost Alerts — What to Watch

1. **Don't enable ElevenLabs voice unnecessarily** — it costs the same as Cartesia but sounds similar for demos
2. **Retell Custom LLM = $0 LLM cost** — always use this in demo mode
3. **Sarvam STT is only needed if Retell transcripts are insufficient** — skip for demo
4. **International calling rates vary** — Indian mobile numbers may have surcharges via Twilio
