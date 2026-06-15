"""streamlit-webrtc audio processor — bridges WebRTC and the Gemini session.

The WebRTC stack sends mic audio as av.AudioFrame objects at 48 kHz (stereo
or mono, float or int16 planar). This module downsamples them to 16 kHz
mono PCM for Gemini. Translated audio arrives pre-upsampled at 48 kHz
(done in gemini_session.py) so recv() only needs to copy from the buffer.

Audio format handling:
  Input (mic → Gemini):  48 kHz any-format → int16 mono → decimate to 16 kHz
  Output (Gemini → browser): 48 kHz int16 mono → match input frame format
"""

import queue

import av
import numpy as np
from streamlit_webrtc import AudioProcessorBase

from src.config import INPUT_RATE, WEBRTC_RATE
from src.gemini_session import GeminiTranslatorSession

# 2 seconds of 48 kHz 16-bit mono — discard stale audio beyond this
_MAX_BUFFER_BYTES: int = WEBRTC_RATE * 2 * 2


def _decimate_to_16k(data: bytes, from_rate: int) -> bytes:
    """Downsample int16 mono PCM to 16 kHz by simple integer decimation.

    Voice energy sits mostly below 8 kHz, well under the 16 kHz Nyquist,
    so the aliasing from skipping anti-aliasing filter is inaudible.
    """
    if from_rate == INPUT_RATE:
        return data
    factor = from_rate // INPUT_RATE  # 48000 // 16000 = 3
    samples = np.frombuffer(data, dtype=np.int16)
    return samples[::factor].tobytes()


def _frame_to_pcm16_mono(frame: av.AudioFrame) -> bytes:
    """Convert av.AudioFrame (any format, any layout) to int16 mono bytes."""
    raw = frame.to_ndarray()

    if frame.format.name == "fltp":
        # Float planar: shape (channels, samples), range -1.0 … 1.0
        mono_f = raw.mean(axis=0)
        return (mono_f * 32767).clip(-32768, 32767).astype(np.int16).tobytes()

    if frame.format.name == "s16p":
        # Int16 planar: shape (channels, samples)
        return raw.mean(axis=0).astype(np.int16).tobytes()

    # s16 interleaved: flatten then mix channels
    flat = raw.flatten().astype(np.int16)
    n_ch = frame.layout.nb_channels
    if n_ch > 1:
        return flat.reshape(-1, n_ch).mean(axis=1).astype(np.int16).tobytes()
    return flat.tobytes()


class AudioTranslatorProcessor(AudioProcessorBase):
    """Plugs into webrtc_streamer to wire the browser mic/speaker to Gemini."""

    def __init__(self, target_language_code: str) -> None:
        self._mic_queue: queue.Queue[bytes] = queue.Queue(maxsize=100)
        # audio_out_queue holds pre-upsampled 48 kHz int16 mono bytes
        self._audio_out_queue: queue.Queue[bytes] = queue.Queue(maxsize=200)
        self.transcript_queue: queue.Queue[dict] = queue.Queue()
        self._out_buffer = bytearray()

        self._session = GeminiTranslatorSession(
            target_language_code=target_language_code,
            mic_queue=self._mic_queue,
            audio_out_queue=self._audio_out_queue,
            transcript_queue=self.transcript_queue,
        )
        self._session.start()

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        """Called by webrtc_streamer for every ~10 ms of audio frame."""

        # --- Input: mic → 16 kHz mono → Gemini ---
        pcm_mono = _frame_to_pcm16_mono(frame)
        pcm_16k = _decimate_to_16k(pcm_mono, frame.sample_rate)
        try:
            self._mic_queue.put_nowait(pcm_16k)
        except queue.Full:
            pass

        # --- Output: drain 48 kHz buffer from Gemini → browser ---
        while not self._audio_out_queue.empty():
            try:
                self._out_buffer.extend(self._audio_out_queue.get_nowait())
            except queue.Empty:
                break

        # Prevent unbounded delay: keep at most 2 s of buffered audio
        if len(self._out_buffer) > _MAX_BUFFER_BYTES:
            del self._out_buffer[: len(self._out_buffer) - _MAX_BUFFER_BYTES]

        # How many 48 kHz int16 mono bytes match this frame's duration
        out_samples = round(frame.samples * WEBRTC_RATE / frame.sample_rate)
        bytes_needed = out_samples * 2

        if len(self._out_buffer) >= bytes_needed:
            out_pcm = bytes(self._out_buffer[:bytes_needed])
            del self._out_buffer[:bytes_needed]
        else:
            out_pcm = bytes(bytes_needed)  # silence until Gemini responds

        # Build output frame matching the input format so WebRTC can encode it
        samples_int16 = np.frombuffer(out_pcm, dtype=np.int16)
        n_ch = frame.layout.nb_channels
        fmt = frame.format.name

        if fmt == "fltp":
            ch_data = samples_int16.astype(np.float32) / 32768.0
            audio_array = np.tile(ch_data, (n_ch, 1))
        elif fmt == "s16p":
            audio_array = np.tile(samples_int16, (n_ch, 1))
        else:
            # s16 interleaved
            audio_array = np.repeat(samples_int16, n_ch).reshape(1, -1)

        out_frame = av.AudioFrame.from_ndarray(audio_array, format=fmt, layout=frame.layout.name)
        out_frame.sample_rate = WEBRTC_RATE
        out_frame.pts = frame.pts
        return out_frame

    def __del__(self) -> None:
        self._session.stop()
