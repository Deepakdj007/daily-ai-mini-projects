"""
Retell Custom LLM WebSocket endpoint.

Retell connects here during a live call and sends transcript turns.
We call Sarvam-M and stream the response back.

Retell WS message types:
  → update_only        : transcript updated, no response needed
  → response_required  : caller finished speaking, respond now
  → reminder_required  : silence timeout, say something

We respond with:
  { response_type: "response", response_id: int, content: str, content_complete: bool, end_call: bool }
"""
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from services.sarvam_service import chat_completion


logger = logging.getLogger(__name__)
router = APIRouter()

SYSTEM_PROMPT = """You are a friendly voice assistant that helps book medical appointments.
You speak both English and Malayalam fluently — respond in whichever language the caller uses.
Keep every response SHORT — 1 to 2 sentences maximum. Be warm and natural.

Your goal is to collect:
1. The caller's name
2. Preferred appointment date and time
3. Reason for the visit (chief complaint)

Once you have all three, confirm the details back and say goodbye."""


@router.websocket("/llm")
async def retell_llm_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    logger.info("Retell LLM WebSocket connected")

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)

            interaction_type = data.get("interaction_type")
            response_id = data.get("response_id", 0)

            # update_only: just a transcript sync, no reply needed
            if interaction_type == "update_only":
                continue

            # Build messages list from transcript
            transcript: list[dict] = data.get("transcript", [])
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for turn in transcript:
                role = "assistant" if turn.get("role") == "agent" else "user"
                messages.append({"role": role, "content": turn.get("content", "")})

            # Add a nudge for reminder_required (silence)
            if interaction_type == "reminder_required":
                messages.append({
                    "role": "user",
                    "content": "[caller is silent — gently prompt them to continue]",
                })

            try:
                reply = await chat_completion(messages)
            except Exception as e:
                logger.error("Sarvam call failed during live call: %s", e)
                reply = "Sorry, I'm having a little trouble. Could you repeat that?"

            response = {
                "response_type": "response",
                "response_id": response_id,
                "content": reply,
                "content_complete": True,
                "end_call": False,
            }
            await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        logger.info("Retell LLM WebSocket disconnected")
    except Exception as e:
        logger.error("Retell LLM WebSocket error: %s", e)
