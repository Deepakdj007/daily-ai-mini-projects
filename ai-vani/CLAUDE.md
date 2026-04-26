# Voice Booking Agent — Claude Code Instructions

## CRITICAL RULES (read before every task)
- NEVER write code until user explicitly says "write" or "code it" or "implement"
- ALWAYS instruct what to do first, then wait
- BEFORE checking any docs, get today's actual date using a tool or bash command
  - Use the date to search for the latest compatible versions
  - Never assume the date — always check it fresh each session
- ALWAYS search latest docs after confirming today's date — never rely on training knowledge alone
- Use latest stable versions of all packages (verify on PyPI on the actual current date)
- If something is paid, say so BEFORE using it

## Project
A generic voice booking agent — books, cancels, reschedules appointments via phone.
Built to be reusable: works for any business (clinic, salon, restaurant, etc).
Stack: LiveKit (voice infra) + Sarvam AI (STT/TTS for Indian languages) + LangGraph (conversation logic) + FastAPI (backend).

## How to get today's date at session start
Run this at the start of every Claude Code session before anything else:
```bash
date +"%B %d, %Y"
```
Use the output as the reference date for all version checks and doc searches.

## Docs to always check (after confirming date)
- LiveKit Agents: https://docs.livekit.io/agents/
- LiveKit Sarvam plugin: https://docs.livekit.io/reference/python/livekit/plugins/sarvam/index.html
- Sarvam STT: https://docs.sarvam.ai/api-reference-docs/endpoints/speech-to-text
- Sarvam TTS: https://docs.sarvam.ai/api-reference-docs/api-guides-tutorials/text-to-speech/overview
- Sarvam changelog (model deprecations): https://docs.sarvam.ai/api-reference-docs/changelog
- LiveKit SIP/Telephony: https://docs.livekit.io/sip/
- LiveKit Python SDK: https://github.com/livekit/python-sdks
- livekit-agents on PyPI: https://pypi.org/project/livekit-agents/
- livekit-plugins-sarvam on PyPI: https://pypi.org/project/livekit-plugins-sarvam/

## What is FREE vs PAID
FREE: LiveKit Cloud dev tier, Sarvam AI free tier
PAID (confirm with user first): Twilio SIP, production LiveKit Cloud beyond free tier
Always check current pricing pages before recommending any paid service.

## Phase plan (never skip phases, each must have showable output)
See PHASES.md

## Architecture
See ARCHITECTURE.md

## Per-phase build rules
See each PHASE_N.md file

