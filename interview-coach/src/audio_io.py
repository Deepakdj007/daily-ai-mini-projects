"""Microphone input and speaker output for the interview coach.

One PyAudio instance owns both directions: a 16 kHz mono input stream
that feeds the Live API and a 24 kHz mono output stream that plays the
coach's spoken feedback.
"""

import pyaudio

from src.config import (
    AUDIO_FORMAT,
    CHANNELS,
    CHUNK_SIZE,
    RECEIVE_SAMPLE_RATE,
    SEND_SAMPLE_RATE,
)


class AudioIO:
    """Owns the mic and speaker streams for the lifetime of a session."""

    def __init__(self) -> None:
        """Initialise PyAudio without opening any streams yet."""
        self._pyaudio = pyaudio.PyAudio()
        self._mic_stream: pyaudio.Stream | None = None
        self._speaker_stream: pyaudio.Stream | None = None

    def open_mic(self) -> pyaudio.Stream:
        """Open the default microphone at the Live API input format."""
        device = self._pyaudio.get_default_input_device_info()
        self._mic_stream = self._pyaudio.open(
            format=AUDIO_FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=int(device["index"]),
            frames_per_buffer=CHUNK_SIZE,
        )
        return self._mic_stream

    def open_speaker(self) -> pyaudio.Stream:
        """Open the default speaker at the Live API output format."""
        self._speaker_stream = self._pyaudio.open(
            format=AUDIO_FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        return self._speaker_stream

    def close(self) -> None:
        """Stop both streams and release the audio device."""
        for stream in (self._mic_stream, self._speaker_stream):
            if stream is not None:
                stream.close()
        self._pyaudio.terminate()
