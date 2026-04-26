"""
Agent tool functions — the 4 callable actions the LLM can invoke.

Each function is a plain async function decorated with @function_tool.
The LLM decides when to call them based on conversation context.

Data contract:
- All inputs/outputs use English for names, ISO-8601 for dates/times.
- The LLM is instructed to transliterate Malayalam names to English before calling.
"""

import logging
import random
import uuid
from datetime import date, datetime, timedelta

from livekit.agents import function_tool

from vani import db

logger = logging.getLogger("vani.tools")


# ---------------------------------------------------------------------------
# Tool 1 — Check if a patient is already registered
# ---------------------------------------------------------------------------

@function_tool
async def check_patient(phone: str) -> dict:
    """
    Check if a patient is already registered by their phone number.

    Args:
        phone: Patient's phone number in E.164 format (e.g. +917012552376).

    Returns:
        dict with 'found' (bool) and 'patient' (dict or None).
    """
    patient = await db.get_patient_by_phone(phone)
    if patient is None:
        logger.info("Patient not found for phone %s", phone)
        return {"found": False, "patient": None}
    logger.info("Patient found: %s", patient.id)
    return {
        "found": True,
        "patient": {
            "id": patient.id,
            "name": patient.name,
            "phone": patient.phone,
            "age": patient.age,
        },
    }


# ---------------------------------------------------------------------------
# Tool 2 — Register a new patient
# ---------------------------------------------------------------------------

@function_tool
async def register_patient(name: str, phone: str, age: int) -> dict:
    """
    Register a new patient in the system.

    Args:
        name:  Patient's full name in English (transliterate from Malayalam if needed).
        phone: Patient's phone number in E.164 format.
        age:   Patient's age in years.

    Returns:
        dict with 'patient_id' and 'name'.
    """
    patient_id = f"P-{uuid.uuid4().hex[:8].upper()}"
    patient = await db.create_patient(
        patient_id=patient_id, name=name, phone=phone, age=age
    )
    logger.info("Registered new patient: %s (%s)", name, patient_id)
    return {"patient_id": patient.id, "name": patient.name}


# ---------------------------------------------------------------------------
# Tool 3 — Get available appointment slots
# ---------------------------------------------------------------------------

@function_tool
async def get_slots(date_str: str) -> dict:
    """
    Get available appointment slots for a given date.

    Args:
        date_str: Date in YYYY-MM-DD format. Use today or a future date.

    Returns:
        dict with 'date' and 'slots' (list of available time strings).
    """
    try:
        target_date = date.fromisoformat(date_str)
    except ValueError:
        return {"error": f"Invalid date format: {date_str}. Use YYYY-MM-DD."}

    if target_date < date.today():
        return {"error": "Cannot book slots in the past."}

    # Generate realistic clinic slots: 9am–5pm, 30-min intervals, randomly drop some
    all_slots = []
    start = datetime.combine(target_date, datetime.min.time()).replace(hour=9)
    for i in range(16):  # 9:00–17:00, 30-min slots
        slot_time = start + timedelta(minutes=30 * i)
        all_slots.append(slot_time.strftime("%Y-%m-%d %H:%M"))

    # Simulate ~60% availability
    random.seed(str(target_date))  # deterministic per date so repeated calls match
    available = [s for s in all_slots if random.random() > 0.4]

    logger.info("Returning %d slots for %s", len(available), date_str)
    return {"date": date_str, "slots": available}


# ---------------------------------------------------------------------------
# Tool 4 — Book an appointment
# ---------------------------------------------------------------------------

@function_tool
async def book_appointment(patient_id: str, slot: str) -> dict:
    """
    Book an appointment for a registered patient.

    Args:
        patient_id: The patient's ID returned from check_patient or register_patient.
        slot:       The exact slot string from get_slots (YYYY-MM-DD HH:MM).

    Returns:
        dict with 'appointment_id', 'patient_id', 'slot', and 'status'.
    """
    appointment_id = f"APT-{uuid.uuid4().hex[:8].upper()}"
    appointment = await db.create_appointment(
        appointment_id=appointment_id,
        patient_id=patient_id,
        slot=slot,
    )
    logger.info("Appointment confirmed: %s for patient %s", appointment_id, patient_id)
    return {
        "appointment_id": appointment.id,
        "patient_id": appointment.patient_id,
        "slot": appointment.slot,
        "status": appointment.status,
    }
