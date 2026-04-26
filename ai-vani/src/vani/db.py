"""
Database layer — SQLite via aiosqlite.

All queries live here. Nothing else in the codebase imports aiosqlite directly.
Schema is created on first run (init_db). The DB file path comes from config.
"""

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncIterator

import aiosqlite

from vani.config import DB_PATH

logger = logging.getLogger("vani.db")

_CREATE_PATIENTS = """
CREATE TABLE IF NOT EXISTS patients (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    phone       TEXT NOT NULL UNIQUE,
    age         INTEGER,
    created_at  TEXT NOT NULL
)
"""

_CREATE_APPOINTMENTS = """
CREATE TABLE IF NOT EXISTS appointments (
    id          TEXT PRIMARY KEY,
    patient_id  TEXT NOT NULL REFERENCES patients(id),
    slot        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'confirmed',
    created_at  TEXT NOT NULL
)
"""


@dataclass(frozen=True)
class Patient:
    id: str
    name: str
    phone: str
    age: int | None
    created_at: str


@dataclass(frozen=True)
class Appointment:
    id: str
    patient_id: str
    slot: str
    status: str
    created_at: str


@asynccontextmanager
async def _connect() -> AsyncIterator[aiosqlite.Connection]:
    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA foreign_keys=ON")
        yield conn


async def init_db() -> None:
    """Create tables if they don't exist. Call once at agent startup."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with _connect() as conn:
        await conn.execute(_CREATE_PATIENTS)
        await conn.execute(_CREATE_APPOINTMENTS)
        await conn.commit()
    logger.info("DB initialised at %s", DB_PATH)


async def get_patient_by_phone(phone: str) -> Patient | None:
    async with _connect() as conn:
        async with conn.execute(
            "SELECT * FROM patients WHERE phone = ?", (phone,)
        ) as cur:
            row = await cur.fetchone()
            if row is None:
                return None
            return Patient(**dict(row))


async def create_patient(
    patient_id: str, name: str, phone: str, age: int | None
) -> Patient:
    now = datetime.utcnow().isoformat()
    async with _connect() as conn:
        await conn.execute(
            "INSERT INTO patients (id, name, phone, age, created_at) VALUES (?, ?, ?, ?, ?)",
            (patient_id, name, phone, age, now),
        )
        await conn.commit()
    logger.info("Patient created: %s (%s)", name, patient_id)
    return Patient(id=patient_id, name=name, phone=phone, age=age, created_at=now)


async def create_appointment(
    appointment_id: str, patient_id: str, slot: str
) -> Appointment:
    now = datetime.utcnow().isoformat()
    async with _connect() as conn:
        await conn.execute(
            "INSERT INTO appointments (id, patient_id, slot, status, created_at) VALUES (?, ?, ?, 'confirmed', ?)",
            (appointment_id, patient_id, slot, now),
        )
        await conn.commit()
    logger.info("Appointment booked: %s → slot %s", patient_id, slot)
    return Appointment(
        id=appointment_id,
        patient_id=patient_id,
        slot=slot,
        status="confirmed",
        created_at=now,
    )
