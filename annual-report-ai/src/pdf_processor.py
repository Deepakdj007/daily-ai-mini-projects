# src/pdf_processor.py
from dataclasses import dataclass
from pathlib import Path

import fitz  # pymupdf

@dataclass
class PageImage:
    page_number: int   # 1-indexed for human readability
    image_bytes: bytes
    width: int
    height: int
    mime_type: str = "image/png"

class PDFProcessor:
    RENDER_ZOOM = 2.0  # 2x = 144 DPI — quality vs payload size sweet spot
    MAX_PAGES = 50     # guard against 500-page document accidents

    def __init__(self, zoom: float = RENDER_ZOOM):
        self.zoom = zoom
        self._matrix = fitz.Matrix(zoom, zoom)

    def load(self, pdf_path: str | Path) -> list[PageImage]:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

        doc = fitz.open(str(path))
        pages = self._render_all_pages(doc)
        doc.close()
        return pages

    def load_bytes(self, pdf_bytes: bytes) -> list[PageImage]:
        """Load from raw bytes — used by the FastAPI upload endpoint."""
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = self._render_all_pages(doc)
        doc.close()
        return pages

    def _render_all_pages(self, doc: fitz.Document) -> list[PageImage]:
        total = min(len(doc), self.MAX_PAGES)
        results = []

        for i in range(total):
            page = doc[i]
            pixmap = page.get_pixmap(matrix=self._matrix)
            img_bytes = pixmap.tobytes("png")

            results.append(PageImage(
                page_number=i + 1,
                image_bytes=img_bytes,
                width=pixmap.width,
                height=pixmap.height,
            ))

        return results