"""Rich terminal layout for the interview coach.

Renders a live dashboard with three panels: session status (question
number, elapsed time, question text), the coach's live transcript, and
the candidate's live transcript. Updated by the coach every 0.5 s.
"""

import time
from dataclasses import dataclass, field

from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text


@dataclass
class DisplayState:
    """All mutable state that the Rich layout reads on each refresh."""

    question_number: int = 0
    total_questions: int = 10
    current_question: str = "Waiting for session to start..."
    coach_transcript: str = ""
    candidate_transcript: str = ""
    session_start: float = field(default_factory=time.time)
    status: str = "connecting"


class CoachDisplay:
    """Manages the Rich Live layout for the duration of a session."""

    def __init__(self, state: DisplayState) -> None:
        """Store shared state; the coach tasks write it, we read it."""
        self._state = state
        self._console = Console()
        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=2,
            screen=False,
        )

    def _elapsed(self) -> str:
        """Return elapsed time as MM:SS string."""
        seconds = int(time.time() - self._state.session_start)
        return f"{seconds // 60:02d}:{seconds % 60:02d}"

    def _render(self) -> Columns:
        """Build the full layout from current DisplayState."""
        s = self._state

        # Left panel: session status
        status_color = {
            "connecting": "yellow",
            "listening": "green",
            "coaching": "cyan",
            "ended": "dim",
        }.get(s.status, "white")

        status_text = Text()
        status_text.append(f"  Status: ", style="bold")
        status_text.append(f"{s.status.upper()}\n", style=f"bold {status_color}")
        status_text.append(f"  Time:   ", style="bold")
        status_text.append(f"{self._elapsed()}\n\n", style="white")
        status_text.append(f"  Q {s.question_number}/{s.total_questions}\n", style="bold magenta")
        status_text.append(f"\n  {s.current_question}", style="italic white")

        left = Panel(
            status_text,
            title="[bold magenta]Interview Coach[/bold magenta]",
            border_style="magenta",
            width=42,
        )

        # Right panel: transcripts
        transcript_text = Text()
        if s.coach_transcript:
            transcript_text.append("Coach\n", style="bold cyan")
            # Show last 600 chars to avoid overflow
            transcript_text.append(s.coach_transcript[-600:], style="cyan")
            transcript_text.append("\n\n")
        if s.candidate_transcript:
            transcript_text.append("You\n", style="bold green")
            transcript_text.append(s.candidate_transcript[-600:], style="green")

        right = Panel(
            transcript_text,
            title="[bold white]Live Transcript[/bold white]",
            border_style="white",
            width=60,
        )

        return Columns([left, right], equal=False)

    def start(self) -> None:
        """Enter the Rich Live context."""
        self._live.start()

    def stop(self) -> None:
        """Exit the Rich Live context."""
        self._live.stop()

    def refresh(self) -> None:
        """Push a fresh render to the terminal."""
        self._live.update(self._render())

    def print_report(self, report: str) -> None:
        """Print the post-session summary report below the live layout."""
        self._console.print(
            Panel(
                report,
                title="[bold yellow]Session Report[/bold yellow]",
                border_style="yellow",
                padding=(1, 2),
            )
        )
