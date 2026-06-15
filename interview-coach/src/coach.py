"""Live AI interview coach — main asyncio agent.

Opens one Gemini Live session and runs five concurrent tasks:
  1. mic → send (candidate's voice to the model)
  2. camera → send (webcam frames to the model)
  3. receive → queue + transcript (model responses inbound)
  4. speaker playback (model's voice out)
  5. display refresh loop (Rich terminal UI)

When the user presses Ctrl+C the session ends and a post-session report
is generated via a separate generate_content call.
"""

import asyncio

from google import genai
from google.genai import types
from google.genai.live import AsyncSession

from src.audio_io import AudioIO
from src.camera import Camera
from src.config import (
    CHUNK_SIZE,
    FRAME_INTERVAL_SECONDS,
    GEMINI_API_KEY,
    INTERVIEW_QUESTIONS,
    KICKOFF_MESSAGE,
    MODEL,
    PREFIX_PADDING_MS,
    SEND_SAMPLE_RATE,
    SILENCE_DURATION_MS,
    SYSTEM_PROMPT,
    VOICE,
)
from src.display import CoachDisplay, DisplayState
from src.session import SessionTranscript

LIVE_CONFIG = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    system_instruction=SYSTEM_PROMPT,
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE)
        )
    ),
    # Capture both sides of the conversation as text for the report.
    input_audio_transcription=types.AudioTranscriptionConfig(),
    output_audio_transcription=types.AudioTranscriptionConfig(),
    # VAD tuned for interview pacing: 3 s silence before the coach speaks,
    # low end sensitivity so thinking pauses don't trigger a response.
    realtime_input_config=types.RealtimeInputConfig(
        automatic_activity_detection=types.AutomaticActivityDetection(
            end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_LOW,
            silence_duration_ms=SILENCE_DURATION_MS,
            prefix_padding_ms=PREFIX_PADDING_MS,
        )
    ),
    context_window_compression=types.ContextWindowCompressionConfig(
        trigger_tokens=25_600,
        sliding_window=types.SlidingWindow(target_tokens=12_800),
    ),
)


class InterviewCoach:
    """Five asyncio tasks sharing one Live API session."""

    def __init__(self) -> None:
        """Set up the API client, devices, shared state, and display."""
        self._client = genai.Client(api_key=GEMINI_API_KEY)
        self._audio = AudioIO()
        self._camera = Camera()
        self._playback_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._session: AsyncSession | None = None

        self._state = DisplayState(total_questions=len(INTERVIEW_QUESTIONS))
        self._display = CoachDisplay(self._state)
        self._transcript = SessionTranscript()

        # How many answers the candidate has completed. Drives the question
        # number, which advances each time an answer finishes.
        self._answers_given: int = 0

    # ------------------------------------------------------------------ tasks

    async def _stream_microphone(self) -> None:
        """Read raw PCM chunks from the mic and send them upstream."""
        mic = self._audio.open_mic()
        mime_type = f"audio/pcm;rate={SEND_SAMPLE_RATE}"
        while True:
            chunk = await asyncio.to_thread(
                mic.read, CHUNK_SIZE, exception_on_overflow=False
            )
            await self._session.send_realtime_input(
                audio=types.Blob(data=chunk, mime_type=mime_type)
            )

    async def _stream_camera(self) -> None:
        """Send one JPEG frame per second — the Live API maximum."""
        while True:
            frame = await asyncio.to_thread(self._camera.read_jpeg_frame)
            if frame is not None:
                await self._session.send_realtime_input(
                    video=types.Blob(data=frame, mime_type="image/jpeg")
                )
            await asyncio.sleep(FRAME_INTERVAL_SECONDS)

    async def _receive_responses(self) -> None:
        """Dispatch inbound messages to the playback queue and transcripts.

        session.receive() yields until the current turn ends, so the outer
        while True restarts it for every new turn. Interrupted responses are
        drained from the playback queue so the coach stops mid-sentence.
        """
        while True:
            async for message in self._session.receive():
                content = message.server_content
                if content is None:
                    continue

                if content.interrupted:
                    self._drain_playback_queue()

                if content.model_turn:
                    for part in content.model_turn.parts:
                        if part.inline_data and part.inline_data.data:
                            self._playback_queue.put_nowait(part.inline_data.data)

                # Coach's spoken words → display + transcript accumulation
                if content.output_transcription and content.output_transcription.text:
                    self._state.coach_transcript += content.output_transcription.text
                    self._state.status = "coaching"

                # Candidate's spoken words → display + transcript accumulation
                if content.input_transcription and content.input_transcription.text:
                    self._state.candidate_transcript += content.input_transcription.text
                    self._state.status = "listening"

                if content.turn_complete:
                    self._flush_turn()

    def _flush_turn(self) -> None:
        """Commit the current turn's speech and advance the question.

        Each turn is one round: the candidate answers (if they spoke), then
        the coach responds. Candidate is added first to keep chronological
        order. A completed answer bumps the question number forward.
        """
        candidate_spoke = bool(self._state.candidate_transcript)

        if candidate_spoke:
            self._transcript.add("Candidate", self._state.candidate_transcript)
            self._state.candidate_transcript = ""
        if self._state.coach_transcript:
            self._transcript.add("Coach", self._state.coach_transcript)
            self._state.coach_transcript = ""

        if candidate_spoke:
            self._answers_given += 1
            idx = min(self._answers_given, len(INTERVIEW_QUESTIONS) - 1)
            self._state.question_number = idx + 1
            self._state.current_question = INTERVIEW_QUESTIONS[idx]

    def _drain_playback_queue(self) -> None:
        """Discard buffered audio after an interruption."""
        while not self._playback_queue.empty():
            self._playback_queue.get_nowait()

    async def _play_audio(self) -> None:
        """Pull audio chunks off the queue and write them to the speaker."""
        speaker = self._audio.open_speaker()
        while True:
            chunk = await self._playback_queue.get()
            await asyncio.to_thread(speaker.write, chunk)

    async def _refresh_display(self) -> None:
        """Push a Rich layout refresh every 0.5 s."""
        while True:
            self._display.refresh()
            await asyncio.sleep(0.5)

    # --------------------------------------------------------------- run loop

    async def run(self) -> None:
        """Connect and run all five tasks until Ctrl+C."""
        self._display.start()
        self._state.status = "connecting"
        self._display.refresh()

        async with self._client.aio.live.connect(
            model=MODEL, config=LIVE_CONFIG
        ) as session:
            self._session = session
            self._state.status = "coaching"
            self._state.question_number = 1
            self._state.current_question = INTERVIEW_QUESTIONS[0]
            self._display.refresh()

            # Prompt the coach to speak first: greet and ask the opening
            # question without waiting for the candidate to say anything.
            await session.send_client_content(
                turns={"role": "user", "parts": [{"text": KICKOFF_MESSAGE}]},
                turn_complete=True,
            )

            async with asyncio.TaskGroup() as group:
                group.create_task(self._stream_microphone())
                group.create_task(self._stream_camera())
                group.create_task(self._receive_responses())
                group.create_task(self._play_audio())
                group.create_task(self._refresh_display())

    def close(self) -> None:
        """Release devices and stop the display."""
        self._state.status = "ended"
        self._display.stop()
        self._camera.close()
        self._audio.close()

    def print_report(self) -> None:
        """Generate and print the post-session summary report."""
        print("\nGenerating your session report...\n")
        report = self._transcript.generate_report()
        self._display.print_report(report)


def main() -> None:
    """Validate the API key, run the coach, then print the report."""
    if not GEMINI_API_KEY:
        raise SystemExit(
            "GEMINI_API_KEY is missing. Copy .env.example to .env and "
            "paste your key from https://aistudio.google.com/apikey"
        )

    coach = InterviewCoach()
    try:
        asyncio.run(coach.run())
    except KeyboardInterrupt:
        pass
    finally:
        coach.close()
        coach.print_report()


if __name__ == "__main__":
    main()
