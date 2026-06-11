"""Real-time vision-and-voice agent on the Gemini Live API.

Opens one WebSocket session and runs four concurrent loops:
microphone audio up, camera frames up, model responses down,
speaker playback. Speak to the agent and show it things —
it answers out loud. Stop with Ctrl+C.
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
    MODEL,
    SEND_SAMPLE_RATE,
    SYSTEM_PROMPT,
    VOICE,
)

LIVE_CONFIG = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    system_instruction=SYSTEM_PROMPT,
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE)
        )
    ),
    output_audio_transcription=types.AudioTranscriptionConfig(),
    # Without compression the session dies when the context fills up.
    # A sliding window keeps it running indefinitely.
    context_window_compression=types.ContextWindowCompressionConfig(
        trigger_tokens=25_600,
        sliding_window=types.SlidingWindow(target_tokens=12_800),
    ),
)


class LiveVisionAgent:
    """Four asyncio tasks sharing one Live API session."""

    def __init__(self) -> None:
        """Set up the API client, devices, and the playback queue."""
        self._client = genai.Client(api_key=GEMINI_API_KEY)
        self._audio = AudioIO()
        self._camera = Camera()
        self._playback_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._session: AsyncSession | None = None

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
        """Queue the model's audio for playback and print its transcript.

        session.receive() yields messages until the current turn ends,
        so the outer loop restarts it for every new turn. When the user
        interrupts the model mid-sentence, the server flags it and any
        unplayed audio is thrown away so the agent stops talking.
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
                if content.output_transcription and content.output_transcription.text:
                    print(content.output_transcription.text, end="", flush=True)
                if content.turn_complete:
                    print()

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

    async def run(self) -> None:
        """Connect the session and run all four loops until Ctrl+C."""
        async with self._client.aio.live.connect(
            model=MODEL, config=LIVE_CONFIG
        ) as session:
            self._session = session
            print(f"Connected to {MODEL} (voice: {VOICE}).")
            print("Talk to the agent and show it things. Ctrl+C to quit.\n")
            async with asyncio.TaskGroup() as group:
                group.create_task(self._stream_microphone())
                group.create_task(self._stream_camera())
                group.create_task(self._receive_responses())
                group.create_task(self._play_audio())

    def close(self) -> None:
        """Release the camera and audio devices."""
        self._camera.close()
        self._audio.close()


def main() -> None:
    """Validate the API key, then run the agent until interrupted."""
    if not GEMINI_API_KEY:
        raise SystemExit(
            "GEMINI_API_KEY is missing. Copy .env.example to .env and "
            "paste your key from https://aistudio.google.com/apikey"
        )
    agent = LiveVisionAgent()
    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        print("\nSession ended.")
    finally:
        agent.close()


if __name__ == "__main__":
    main()
