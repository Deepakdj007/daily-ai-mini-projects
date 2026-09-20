"""Every pixel operation: resize, hash, PDF page to PNG, CT slice to PNG, overlays.

Inputs:  image files, PDF files, NIfTI volumes
Outputs: PNG bytes sized for the model, sha256 digests, annotated PNG files

The model sees 896x896 whatever it is given, so every image is brought to that
edge here and the hash is taken of the bytes that leave this module. Nothing
downstream ever hashes a source file: a source file is not what the model saw.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src import config
from src.types import Box


def sha256(data: bytes) -> str:
    """The digest of exactly these bytes."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _to_png(image: Image.Image) -> bytes:
    """Encode losslessly. JPEG would make the hash depend on encoder settings."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def prepare(path: Path, edge: int = config.IMAGE_EDGE) -> bytes:
    """Load, convert to RGB and resize so the long edge is `edge`.

    `edge` is a parameter so two images in one message can differ by a pixel:
    Ollama's tokenizer can merge adjacent same-size images into video frames
    (ollama#17321), and a one-pixel difference is the cheapest insurance.
    """
    image = Image.open(path).convert("RGB")
    scale = edge / max(image.size)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return _to_png(image.resize(size, Image.LANCZOS))


def prepare_bytes(data: bytes, edge: int = config.IMAGE_EDGE) -> bytes:
    """Same as `prepare`, for images that were never on disk."""
    image = Image.open(io.BytesIO(data)).convert("RGB")
    scale = edge / max(image.size)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return _to_png(image.resize(size, Image.LANCZOS))


def pdf_page_png(pdf_path: Path, page: int = 0, dpi: int = config.LAB_DPI) -> bytes:
    """Rasterise one PDF page. pymupdf ships its own renderer - no poppler needed."""
    import pymupdf

    document = pymupdf.open(pdf_path)
    pixmap = document[page].get_pixmap(dpi=dpi)
    data = pixmap.tobytes("png")
    document.close()
    return prepare_bytes(data)


def burn_text(text: str, edge: int = config.IMAGE_EDGE) -> bytes:
    """A canary frame: a plain grey square with `text` printed large.

    If the model can read this sighted and cannot blind, the image reached it.
    A mis-sent image produces a confident answer with a 200 OK and no error,
    which is why this exists.
    """
    image = Image.new("RGB", (edge, edge), (40, 40, 40))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=edge // 9)
    box = draw.textbbox((0, 0), text, font=font)
    position = ((edge - (box[2] - box[0])) // 2, (edge - (box[3] - box[1])) // 2)
    draw.text(position, text, fill=(235, 235, 235), font=font)
    return _to_png(image)


def draw_boxes(png: bytes, boxes: Sequence[Box], out_path: Path) -> Path:
    """Paint normalised boxes onto the image the model was shown, and save it."""
    image = Image.open(io.BytesIO(png)).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    font = ImageFont.load_default(size=max(14, width // 40))
    for box in boxes:
        rect = (box.x0 * width, box.y0 * height, box.x1 * width, box.y1 * height)
        draw.rectangle(rect, outline=(255, 64, 64), width=max(2, width // 300))
        draw.text((rect[0] + 4, max(0, rect[1] - font.size - 4)), box.label,
                  fill=(255, 64, 64), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, format="PNG")
    return out_path


def _window(slice_hu: np.ndarray, low: int, high: int) -> np.ndarray:
    """Map one HU range to 0..255, clipping outside it."""
    scaled = (np.clip(slice_hu, low, high) - low) / (high - low)
    return (scaled * 255).astype(np.uint8)


def ct_slice_png(slice_hu: np.ndarray, edge: int = config.IMAGE_EDGE) -> bytes:
    """One axial slice as the paper's three-window RGB image, resized to `edge`."""
    channels = [_window(slice_hu, low, high) for low, high in config.CT_WINDOWS_HU]
    image = Image.fromarray(np.stack(channels, axis=-1), mode="RGB")
    return _to_png(image.resize((edge, edge), Image.LANCZOS))


def ct_slices(nii_path: Path, n: int) -> list[bytes]:
    """`n` axial slices sampled equidistantly through a NIfTI volume."""
    import nibabel as nib

    volume = nib.as_closest_canonical(nib.load(nii_path))
    data = np.asarray(volume.get_fdata(), dtype=np.float32)
    depth = data.shape[2]
    picks = np.linspace(0, depth - 1, num=min(n, depth)).round().astype(int)
    # RAS orientation: rotate so anterior is up when the array is shown as an image.
    return [ct_slice_png(np.rot90(data[:, :, k])) for k in picks]


if __name__ == "__main__":
    frame = burn_text("CASE ID KQ-7741")
    assert Image.open(io.BytesIO(frame)).size == (config.IMAGE_EDGE, config.IMAGE_EDGE)
    assert sha256(frame) == sha256(burn_text("CASE ID KQ-7741")), "canary must be deterministic"
    demo = config.DEMO_DIR / "longitudinal_before.png"
    if demo.exists():
        png = prepare(demo)
        print(f"prepared {demo.name}: {Image.open(io.BytesIO(png)).size}, {sha256(png)[:20]}...")
        out = draw_boxes(png, [Box(0.35, 0.3, 0.7, 0.75, "cardiac silhouette")],
                         config.OVERLAYS_DIR / "selftest.png")
        print(f"overlay written to {out}")
    ct = config.DEMO_DIR / "CT_Abdo.nii.gz"
    if ct.exists():
        slices = ct_slices(ct, 4)
        print(f"{len(slices)} CT slices, first {len(slices[0]):,} bytes")
    print("OK - images")
