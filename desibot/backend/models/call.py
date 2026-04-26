import json
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict


class CallStatus(str, Enum):
    initiated = "initiated"
    in_progress = "in_progress"
    completed = "completed"
    failed = "failed"
    ended = "ended"


class TranscriptTurn(BaseModel):
    role: str  # "agent" | "user"
    content: str


class ExtractedData(BaseModel):
    person_name: Optional[str] = None
    phone_confirmed: bool = False
    appointment_date: Optional[str] = None
    appointment_time: Optional[str] = None
    chief_complaint: Optional[str] = None
    callback_requested: bool = False
    language_spoken: str = "English"
    call_outcome: Optional[str] = None  # confirmed | rescheduled | declined | no_answer | incomplete


class CallRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    call_id: str
    to_number: str
    from_number: str
    agent_id: str
    status: CallStatus = CallStatus.initiated
    schema_type: str = "appointment"
    transcript: Optional[str] = None
    transcript_object: Optional[list[TranscriptTurn]] = None
    extracted_data: Optional[ExtractedData] = None
    metadata: Optional[dict] = None
    created_at: datetime
    updated_at: datetime
    disconnection_reason: Optional[str] = None


class InitiateCallRequest(BaseModel):
    to_number: str
    schema_type: str = "appointment"
    metadata: Optional[dict] = None
    dynamic_variables: Optional[dict] = None


class CallRecordRow(BaseModel):
    """Raw row from SQLite — transcript_object and extracted_data are JSON strings."""
    model_config = ConfigDict(from_attributes=True)

    call_id: str
    to_number: str
    from_number: str
    agent_id: str
    status: str
    schema_type: str
    transcript: Optional[str]
    transcript_object: Optional[str]  # JSON string
    extracted_data: Optional[str]     # JSON string
    metadata: Optional[str]           # JSON string
    created_at: str
    updated_at: str
    disconnection_reason: Optional[str]

    def to_call_record(self) -> CallRecord:
        return CallRecord(
            call_id=self.call_id,
            to_number=self.to_number,
            from_number=self.from_number,
            agent_id=self.agent_id,
            status=CallStatus(self.status),
            schema_type=self.schema_type,
            transcript=self.transcript,
            transcript_object=(
                [TranscriptTurn(**t) for t in json.loads(self.transcript_object)]
                if self.transcript_object else None
            ),
            extracted_data=(
                ExtractedData(**json.loads(self.extracted_data))
                if self.extracted_data else None
            ),
            metadata=json.loads(self.metadata) if self.metadata else None,
            created_at=datetime.fromisoformat(self.created_at),
            updated_at=datetime.fromisoformat(self.updated_at),
            disconnection_reason=self.disconnection_reason,
        )
