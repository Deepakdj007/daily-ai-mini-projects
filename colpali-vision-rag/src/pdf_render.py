"""Turn PDF pages into images — the only "parsing" this RAG ever does.

There is no OCR and no text extraction. ColPali reads each page as a picture,
so our job is simply: PDF -> one PIL image per page -> saved PNG on disk.
We keep the PNGs because the answer step later feeds those exact images to Gemini.
"""

from pathlib import Path

from pdf2image import convert_from_path
from PIL import Image

from src.config import PAGE_IMAGES_DIR, POPPLER_PATH, RENDER_DPI


def pdf_to_images(pdf_path: Path, dpi: int = RENDER_DPI) -> list[Image.Image]:
    """Render every page of a PDF to an RGB PIL image.

    Args:
        pdf_path: Path to the source PDF.
        dpi: Render resolution. Higher = sharper but slower to embed.

    Returns:
        One PIL image per page, in page order.
    """
    poppler = POPPLER_PATH if POPPLER_PATH else None
    return convert_from_path(str(pdf_path), dpi=dpi, fmt="RGB", poppler_path=poppler)


def save_page_image(image: Image.Image, pdf_name: str, page_number: int) -> Path:
    """Save one rendered page as a PNG and return its path.

    The filename encodes the source PDF and 1-based page number so retrieval
    results can point straight back at the picture Gemini should look at.
    """
    PAGE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PAGE_IMAGES_DIR / f"{Path(pdf_name).stem}_page_{page_number}.png"
    image.save(out_path, format="PNG")
    return out_path
