"""Background Gemini Live translate session.

Runs a single Gemini Live WebSocket session in its own asyncio event loop
on a daemon thread. Three thread-safe queues connect it to the audio
processor running on the WebRTC thread:

  mic_queue       — caller puts raw 16 kHz PCM bytes; we forward to Gemini
  audio_out_queue — we put 48 kHz PCM bytes (upsampled from Gemini's 24 kHz)
  transcript_queue — we put {"type": "input"|"output", "text": str} dicts

Upsampling happens here (24 kHz → 48 kHz) so the hot recv() path in the
WebRTC thread just does a memcpy into the buffer — no resampling on every
10 ms frame callback.
"""

import asyncio
import queue
import threading

import numpy as np
from google import genai
from google.genai import types

from src.config import GEMINI_API_KEY, GEMINI_MODEL, INPUT_RATE


def _upsample_24k_to_48k(data: bytes) -> bytes:
    """Upsample Gemini's 24 kHz int16 PCM to 48 kHz by 2× sample repetition.

    Using np.repeat (exact integer ratio) avoids the filter boundary
    transients that resample_poly produces on short chunks.
    """
    samples = np.frombuffer(data, dtype=np.int16)
    return np.repeat(samples, 2).tobytes()


class GeminiTranslatorSession:
    """One Gemini Live translate session running on a background thread."""

    def __init__(
        self,
        target_language_code: str,
        mic_queue: queue.Queue,
        audio_out_queue: queue.Queue,
        transcript_queue: queue.Queue,
    ) -> None:
        self._target_lang = target_language_code
        self._mic_queue = mic_queue
        self._audio_out_queue = audio_out_queue
        self._transcript_queue = transcript_queue
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Spawn the daemon thread and begin the session."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the session to stop and wait up to 3 s for the thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)

    # ------------------------------------------------------------------
    # Internal — runs entirely on the background thread
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Entry point for the daemon thread; owns its own event loop."""
        try:
            asyncio.run(self._session_loop())
        except Exception as exc:
            self._transcript_queue.put_nowait({"type": "error", "text": str(exc)})

    async def _session_loop(self) -> None:
        """Open one Live session and run sender + receiver concurrently."""
        client = genai.Client(api_key=GEMINI_API_KEY)
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            translation_config=types.TranslationConfig(
                target_language_code=self._target_lang,
                echo_target_language=False,
            ),
        )
        async with client.aio.live.connect(model=GEMINI_MODEL, config=config) as session:
            await asyncio.gather(
                self._mic_sender(session),
                self._receiver(session),
            )

    async def _mic_sender(self, session) -> None:
        """Drain mic_queue and stream each chunk to Gemini."""
        loop = asyncio.get_event_loop()
        mime = f"audio/pcm;rate={INPUT_RATE}"
        while not self._stop_event.is_set():
            try:
                chunk: bytes = await loop.run_in_executor(
                    None, self._mic_queue.get, True, 0.1
                )
            except queue.Empty:
                continue
            await session.send_realtime_input(
                audio=types.Blob(data=chunk, mime_type=mime)
            )

    async def _receiver(self, session) -> None:
        """Route translated audio and transcripts to their output queues."""
        while not self._stop_event.is_set():
            async for msg in session.receive():
                if self._stop_event.is_set():
                    return
                sc = msg.server_content
                if sc is None:
                    continue

                # Translated audio: upsample 24 kHz → 48 kHz before queuing
                # so the WebRTC recv() thread only does buffer copies, not resampling
                if sc.model_turn:
                    for part in sc.model_turn.parts:
                        if part.inline_data and part.inline_data.data:
                            chunk_48k = _upsample_24k_to_48k(part.inline_data.data)
                            try:
                                self._audio_out_queue.put_nowait(chunk_48k)
                            except queue.Full:
                                pass

                if sc.input_transcription and sc.input_transcription.text:
                    self._transcript_queue.put_nowait(
                        {"type": "input", "text": sc.input_transcription.text}
                    )

                if sc.output_transcription and sc.output_transcription.text:
                    self._transcript_queue.put_nowait(
                        {"type": "output", "text": sc.output_transcription.text}
                    )
