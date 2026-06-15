"""Interview session transcript accumulator and post-session report.

Collects everything said during the session (candidate answers + coach
feedback) and generates a structured written summary via a single
generate_content call after the Live session ends.
"""

import time
from dataclasses import dataclass, field

from google import genai

from src.config import GEMINI_API_KEY, REPORT_MODEL


@dataclass
class Turn:
    """One exchange: a speaker label and the text of what they said."""

    speaker: str  # "Candidate" or "Coach"
    text: str
    timestamp: float = field(default_factory=time.time)


class SessionTranscript:
    """Accumulates turns during the Live session."""

    def __init__(self) -> None:
        """Start with an empty turn list."""
        self._turns: list[Turn] = []
        self._start_time: float = time.time()

    def add(self, speaker: str, text: str) -> None:
        """Append a completed turn."""
        text = text.strip()
        if text:
            self._turns.append(Turn(speaker=speaker, text=text))

    def duration_minutes(self) -> float:
        """Return how long the session ran in minutes."""
        return (time.time() - self._start_time) / 60

    def to_text(self) -> str:
        """Serialise the full transcript as labelled turns."""
        lines = []
        for t in self._turns:
            lines.append(f"[{t.speaker}]: {t.text}")
        return "\n\n".join(lines)

    def generate_report(self) -> str:
        """Call generate_content to produce a structured post-session report.

        Uses a standard text model (not the Live model, which is WebSocket
        only) so the report is plain text from a single REST call.
        """
        transcript_text = self.to_text()
        if not transcript_text:
            return "No transcript captured — check your microphone and API key."

        client = genai.Client(api_key=GEMINI_API_KEY)
        prompt = f"""You are an expert interview coach reviewing a mock interview transcript.

Transcript:
{transcript_text}

Write a structured post-session report with these sections:

OVERALL SCORE: X/10 (one sentence rationale)

TOP 3 STRENGTHS:
- [specific observation from transcript]
- [specific observation from transcript]
- [specific observation from transcript]

TOP 3 AREAS TO IMPROVE:
- [specific observation with concrete fix]
- [specific observation with concrete fix]
- [specific observation with concrete fix]

FILLER WORD SUMMARY:
Count any filler words (um, uh, like, you know, basically, actually) from the candidate's turns and list totals.

ONE PRIORITY ACTION:
The single most important thing to practise before the next interview.

Keep the report concise and specific. No vague advice. Name exact moments from the transcript."""

        response = client.models.generate_content(
            model=REPORT_MODEL,
            contents=prompt,
        )
        return response.text or "Report generation failed — empty response."
