"""Ingest CLI: PDF pages -> page PNGs -> ColPali multivectors -> Qdrant.

Run this once per document set. It rebuilds the collection from scratch so the
store always matches the PDFs you pass in.

Usage:
    PYTHONPATH=. uv run python src/ingest.py pdfs/report.pdf
    PYTHONPATH=. uv run python src/ingest.py            # all PDFs in pdfs/
"""

import sys
from pathlib import Path

from src.config import PDFS_DIR
from src.embedder import embed_image
from src.pdf_render import pdf_to_images, save_page_image
from src.vector_store import close_client, ensure_collection, upsert_page


def ingest_pdf(pdf_path: Path, start_id: int) -> int:
    """Render, embed, and store every page of one PDF.

    Args:
        pdf_path: The PDF to ingest.
        start_id: First Qdrant point id to use (ids must be unique across PDFs).

    Returns:
        The next free point id after this PDF's pages.
    """
    print(f"\nRendering {pdf_path.name} ...")
    pages = pdf_to_images(pdf_path)
    print(f"  {len(pages)} pages")

    for offset, page in enumerate(pages):
        page_number = offset + 1
        image_path = save_page_image(page, pdf_path.name, page_number)
        multivector = embed_image(page)
        upsert_page(
            point_id=start_id + offset,
            multivector=multivector,
            pdf_name=pdf_path.name,
            page_number=page_number,
            image_path=image_path,
        )
        print(f"  embedded + stored page {page_number}")

    return start_id + len(pages)


def main(pdf_args: list[str]) -> None:
    """Resolve which PDFs to ingest, then build the Qdrant collection fresh."""
    if pdf_args:
        pdfs = [Path(p) for p in pdf_args]
    else:
        pdfs = sorted(PDFS_DIR.glob("*.pdf"))

    if not pdfs:
        print(f"No PDFs found. Put a PDF in {PDFS_DIR} or pass a path.")
        sys.exit(1)

    ensure_collection(reset=True)

    next_id = 0
    try:
        for pdf in pdfs:
            if not pdf.exists():
                print(f"Skipping missing file: {pdf}")
                continue
            next_id = ingest_pdf(pdf, next_id)
    finally:
        close_client()

    print(f"\nDone. Indexed {next_id} pages into Qdrant.")


if __name__ == "__main__":
    main(sys.argv[1:])
