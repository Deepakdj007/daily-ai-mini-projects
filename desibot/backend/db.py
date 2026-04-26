import json
import logging
from datetime import datetime, UTC
from typing import Optional

import aiosqlite

from config import settings
from models.call import CallRecord, CallRecordRow, CallStatus, ExtractedData, TranscriptTurn


logger = logging.getLogger(__name__)

_db_path: str = settings.DATABASE_URL


async def init_db() -> None:
    """Create tables if they don't exist. Called at app startup."""
    async with aiosqlite.connect(_db_path) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS calls (
                call_id            TEXT PRIMARY KEY,
                to_number          TEXT NOT NULL,
                from_number        TEXT NOT NULL,
                agent_id           TEXT NOT NULL,
                status             TEXT NOT NULL DEFAULT 'initiated',
                schema_type        TEXT NOT NULL DEFAULT 'appointment',
                transcript         TEXT,
                transcript_object  TEXT,
                extracted_data     TEXT,
                metadata           TEXT,
                created_at         TEXT NOT NULL,
                updated_at         TEXT NOT NULL,
                disconnection_reason TEXT
            )
        """)
        await db.commit()
    logger.info("Database initialised at %s", _db_path)


async def insert_call(record: CallRecord) -> None:
    async with aiosqlite.connect(_db_path) as db:
        await db.execute(
            """
            INSERT INTO calls
                (call_id, to_number, from_number, agent_id, status, schema_type,
                 transcript, transcript_object, extracted_data, metadata,
                 created_at, updated_at, disconnection_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.call_id,
                record.to_number,
                record.from_number,
                record.agent_id,
                record.status.value,
                record.schema_type,
                record.transcript,
                (
                    json.dumps([t.model_dump() for t in record.transcript_object])
                    if record.transcript_object else None
                ),
                record.extracted_data.model_dump_json() if record.extracted_data else None,
                json.dumps(record.metadata) if record.metadata else None,
                record.created_at.isoformat(),
                record.updated_at.isoformat(),
                record.disconnection_reason,
            ),
        )
        await db.commit()


async def update_call_status(
    call_id: str,
    status: CallStatus,
    disconnection_reason: Optional[str] = None,
) -> None:
    now = datetime.now(UTC).isoformat()
    async with aiosqlite.connect(_db_path) as db:
        await db.execute(
            """
            UPDATE calls
            SET status = ?, updated_at = ?, disconnection_reason = ?
            WHERE call_id = ?
            """,
            (status.value, now, disconnection_reason, call_id),
        )
        await db.commit()


async def update_call_transcript(
    call_id: str,
    transcript: str,
    transcript_object: Optional[list[TranscriptTurn]] = None,
) -> None:
    now = datetime.now(UTC).isoformat()
    async with aiosqlite.connect(_db_path) as db:
        await db.execute(
            """
            UPDATE calls
            SET transcript = ?, transcript_object = ?, updated_at = ?
            WHERE call_id = ?
            """,
            (
                transcript,
                (
                    json.dumps([t.model_dump() for t in transcript_object])
                    if transcript_object else None
                ),
                now,
                call_id,
            ),
        )
        await db.commit()


async def update_call_extracted_data(call_id: str, extracted: ExtractedData) -> None:
    now = datetime.now(UTC).isoformat()
    async with aiosqlite.connect(_db_path) as db:
        await db.execute(
            "UPDATE calls SET extracted_data = ?, updated_at = ? WHERE call_id = ?",
            (extracted.model_dump_json(), now, call_id),
        )
        await db.commit()


async def get_call(call_id: str) -> Optional[CallRecord]:
    async with aiosqlite.connect(_db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM calls WHERE call_id = ?", (call_id,)
        ) as cursor:
            row = await cursor.fetchone()
    if row is None:
        return None
    return CallRecordRow(**dict(row)).to_call_record()


async def list_calls(limit: int = 100, offset: int = 0) -> list[CallRecord]:
    async with aiosqlite.connect(_db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM calls ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ) as cursor:
            rows = await cursor.fetchall()
    return [CallRecordRow(**dict(r)).to_call_record() for r in rows]
