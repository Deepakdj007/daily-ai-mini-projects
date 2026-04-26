# EXTRACTION_SCHEMA.md

Defines the variables to extract from call transcripts using Sarvam 30B (free LLM).

---

## Default Schema (Appointment Booking)

```json
{
  "person_name": "string | null",
  "phone_confirmed": "boolean",
  "appointment_date": "string | null (ISO date)",
  "appointment_time": "string | null",
  "chief_complaint": "string | null",
  "callback_requested": "boolean",
  "language_spoken": "string (e.g. English, Hindi, Malayalam)",
  "call_outcome": "confirmed | rescheduled | declined | no_answer | incomplete"
}
```

---

## Extraction Prompt Template

Used in `services/extractor.py`:

```
You are a data extraction assistant. Given a call transcript, extract the following fields as a JSON object.
Return ONLY valid JSON, no explanation.

Fields to extract:
- person_name: The name of the person spoken to (string or null)
- phone_confirmed: Did they confirm the phone number is correct? (boolean)
- appointment_date: Any date mentioned for an appointment (ISO 8601 string or null)
- appointment_time: Any time mentioned (HH:MM string or null)
- chief_complaint: Main reason for the appointment in 1-2 sentences (string or null)
- callback_requested: Did they ask to be called back? (boolean)
- language_spoken: Primary language used in the call (string)
- call_outcome: One of: confirmed, rescheduled, declined, no_answer, incomplete

Transcript:
{{TRANSCRIPT}}
```

---

## Adding Custom Schemas

For different call types (lead qualification, surveys, etc.), add new schemas here and reference them by `schema_type` in the `InitiateCallRequest` body.

### Lead Qualification Schema
```json
{
  "lead_name": "string | null",
  "interested": "boolean",
  "budget_range": "string | null",
  "timeline": "string | null",
  "follow_up_date": "string | null",
  "call_outcome": "hot_lead | warm_lead | cold_lead | not_interested | no_answer"
}
```
