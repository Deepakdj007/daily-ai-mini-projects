import logging

from fastapi import APIRouter, HTTPException, Request
from retell import Retell

import db
from config import settings
from models.call import CallStatus, TranscriptTurn
from services.extractor import extract_variables


logger = logging.getLogger(__name__)
router = APIRouter()

_retell_client = Retell(api_key=settings.RETELL_API_KEY)


@router.post("/retell")
async def retell_webhook(request: Request) -> dict:
    raw_body = await request.body()

    # Signature verification (skip if no secret configured)
    if settings.RETELL_WEBHOOK_SECRET:
        signature = request.headers.get("X-Retell-Signature", "")
        valid = _retell_client.verify(
            raw_body.decode("utf-8"),
            api_key=settings.RETELL_API_KEY,
            signature=signature,
        )
        if not valid:
            raise HTTPException(status_code=401, detail="Invalid webhook signature")

    import json
    payload = json.loads(raw_body)
    event = payload.get("event")
    call_data = payload.get("call", {})
    call_id = call_data.get("call_id")

    if not call_id:
        logger.warning("Webhook payload missing call_id: %s", payload)
        return {"ok": True}

    logger.info("Retell webhook: event=%s call_id=%s", event, call_id)

    if event == "call_started":
        await db.update_call_status(call_id, CallStatus.in_progress)

    elif event == "transcript_updated":
        transcript_text = call_data.get("transcript", "")
        raw_turns = call_data.get("transcript_object", [])
        turns = [TranscriptTurn(role=t["role"], content=t["content"]) for t in raw_turns]
        await db.update_call_transcript(call_id, transcript_text, turns)

    elif event == "call_ended":
        reason = call_data.get("disconnection_reason")
        await db.update_call_status(call_id, CallStatus.ended, disconnection_reason=reason)

    elif event == "call_analyzed":
        # Full transcript + analysis ready — run extraction
        transcript_text = call_data.get("transcript", "")
        raw_turns = call_data.get("transcript_object", [])
        turns = [TranscriptTurn(role=t["role"], content=t["content"]) for t in raw_turns]

        await db.update_call_transcript(call_id, transcript_text, turns)
        await db.update_call_status(call_id, CallStatus.completed)

        # Fetch schema_type from our DB so we use the right extraction schema
        record = await db.get_call(call_id)
        schema_type = record.schema_type if record else "appointment"

        extracted = await extract_variables(transcript_text, schema_type)
        if extracted:
            await db.update_call_extracted_data(call_id, extracted)
            logger.info("Extraction complete for %s: %s", call_id, extracted)

    return {"ok": True}
