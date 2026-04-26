# agent/image.py
import base64
from pathlib import Path

# Map file extensions to MIME types.
# The API needs to know what kind of image it is receiving.
SUPPORTED_FORMATS = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def load_from_path(image_path: str) -> tuple[str, str]:
    """
    Load an image from a file path on disk.
    Returns (base64_string, media_type).
    Use this for scripts and programmatic testing.
    """
    path = Path(image_path)
    media_type = SUPPORTED_FORMATS.get(path.suffix.lower())

    if not media_type:
        raise ValueError(
            f"Unsupported format: '{path.suffix}'. "
            f"Supported formats: {list(SUPPORTED_FORMATS.keys())}"
        )

    with open(path, "rb") as f:
        image_bytes = f.read()

    return _encode(image_bytes), media_type


def load_from_bytes(image_bytes: bytes, filename: str) -> tuple[str, str]:
    """
    Load an image from raw bytes.
    Use this when Streamlit passes the uploaded file directly —
    Streamlit gives you bytes in memory, not a file path on disk.
    Returns (base64_string, media_type).
    """
    suffix = Path(filename).suffix.lower()
    media_type = SUPPORTED_FORMATS.get(suffix)

    if not media_type:
        raise ValueError(
            f"Unsupported format: '{suffix}'. "
            f"Supported formats: {list(SUPPORTED_FORMATS.keys())}"
        )

    return _encode(image_bytes), media_type


def _encode(image_bytes: bytes) -> str:
    """
    Convert raw bytes to a base64 string.
    The underscore prefix marks this as a private helper —
    not meant to be imported or called from outside this module.
    """
    return base64.b64encode(image_bytes).decode("utf-8")