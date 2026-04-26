import logging
from datetime import datetime, UTC

from fastapi import APIRouter, HTTPException

import db
from models.call import CallRecord, CallStatus, InitiateCallRequest
from services import retell_service


logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/initiate", response_model=CallRecord, status_code=201)
async def initiate_call(body: InitiateCallRequest) -> CallRecord:
    try:
        retell_call = await retell_service.create_phone_call(
            to_number=body.to_number,
            metadata=body.metadata,
            dynamic_variables=body.dynamic_variables,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to create Retell call: %s", e)
        raise HTTPException(status_code=502, detail="Failed to initiate call via Retell")

    from config import settings
    now = datetime.now(UTC)
    record = CallRecord(
        call_id=retell_call.call_id,
        to_number=body.to_number,
        from_number=settings.RETELL_FROM_NUMBER,
        agent_id=settings.RETELL_AGENT_ID,
        status=CallStatus.initiated,
        schema_type=body.schema_type,
        metadata=body.metadata,
        created_at=now,
        updated_at=now,
    )
    await db.insert_call(record)
    return record


@router.get("/", response_model=list[CallRecord])
async def list_calls(limit: int = 50, offset: int = 0) -> list[CallRecord]:
    return await db.list_calls(limit=limit, offset=offset)


@router.get("/{call_id}", response_model=CallRecord)
async def get_call(call_id: str) -> CallRecord:
    record = await db.get_call(call_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Call not found")
    return record
