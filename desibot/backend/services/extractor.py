import json
import logging
from typing import Optional

from models.call import ExtractedData
from services.sarvam_service import chat_completion


logger = logging.getLogger(__name__)

APPOINTMENT_PROMPT = """You are a data extraction assistant. Given a call transcript, extract the following fields as a JSON object.
Return ONLY valid JSON, no explanation, no markdown.

Fields to extract:
- person_name: The name of the person spoken to (string or null)
- phone_confirmed: Did they confirm the phone number is correct? (boolean)
- appointment_date: Any date mentioned for an appointment (ISO 8601 string or null)
- appointment_time: Any time mentioned (HH:MM string or null)
- chief_complaint: Main reason for the appointment in 1-2 sentences (string or null)
- callback_requested: Did they ask to be called back? (boolean)
- language_spoken: Primary language used in the call (string, e.g. English, Hindi, Malayalam)
- call_outcome: One of: confirmed, rescheduled, declined, no_answer, incomplete

Transcript:
{transcript}"""

LEAD_PROMPT = """You are a data extraction assistant. Given a call transcript, extract the following fields as a JSON object.
Return ONLY valid JSON, no explanation, no markdown.

Fields to extract:
- person_name: Name of the lead (string or null)
- phone_confirmed: Did they confirm the phone number? (boolean)
- interested: Is the lead interested in the product/service? (boolean)
- callback_requested: Did they ask to be called back? (boolean)
- language_spoken: Primary language used in the call (string)
- call_outcome: One of: hot_lead, warm_lead, cold_lead, not_interested, no_answer, incomplete

Transcript:
{transcript}"""

PROMPTS = {
    "appointment": APPOINTMENT_PROMPT,
    "lead": LEAD_PROMPT,
}


async def extract_variables(
    transcript: str,
    schema_type: str = "appointment",
) -> Optional[ExtractedData]:
    """Send transcript to Sarvam-M and parse structured extraction result."""
    if not transcript or not transcript.strip():
        logger.warning("Empty transcript — skipping extraction")
        return None

    prompt_template = PROMPTS.get(schema_type, APPOINTMENT_PROMPT)
    prompt = prompt_template.format(transcript=transcript)

    messages = [{"role": "user", "content": prompt}]

    try:
        raw = await chat_completion(messages)
        # strip markdown fences if model adds them
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        data = json.loads(raw)
        return ExtractedData(**data)
    except json.JSONDecodeError as e:
        logger.error("Sarvam returned non-JSON: %s | raw: %.200s", e, raw)
        return None
    except Exception as e:
        logger.error("Extraction failed: %s", e)
        return None
