"""Webcam capture for the live vision agent.

Wraps OpenCV's VideoCapture: grabs one frame at a time, downsizes it
to the Live API's recommended resolution, and returns it as JPEG bytes
ready to stream.
"""

import io

import cv2
from PIL import Image

from src.config import JPEG_QUALITY, MAX_FRAME_SIZE


class Camera:
    """One webcam, opened once, read one JPEG frame at a time."""

    def __init__(self, device_index: int = 0) -> None:
        """Open the webcam. Raises RuntimeError if no camera is found."""
        self._capture = cv2.VideoCapture(device_index)
        if not self._capture.isOpened():
            raise RuntimeError(
                f"No camera found at index {device_index}. "
                "Close other apps using the webcam and try again."
            )

    def read_jpeg_frame(self) -> bytes | None:
        """Grab one frame and return it as JPEG bytes, or None on failure.

        OpenCV delivers frames in BGR order; Pillow expects RGB, so the
        channels are flipped before encoding. thumbnail() preserves the
        aspect ratio while capping the longest side at 768 px.
        """
        ok, frame = self._capture.read()
        if not ok:
            return None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        image.thumbnail(MAX_FRAME_SIZE)

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
        return buffer.getvalue()

    def close(self) -> None:
        """Release the webcam so other apps can use it."""
        self._capture.release()
