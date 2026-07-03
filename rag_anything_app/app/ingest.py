"""Document ingestion: parse a folder of documents into the knowledge graph.

Uses RAG-Anything's ``process_folder_complete`` for the common case, plus a
per-file fallback that skips (and logs) any document that fails to parse so one
bad file never crashes the whole batch.

Inputs: a folder path (defaults to ./documents).
Output: parsed content stored in ./rag_storage, ready to query.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .config import DOCUMENTS_DIR, OUTPUT_DIR, get_rag

logger = logging.getLogger("rag_anything_app")

# Extensions we attempt. Office formats (.docx/.pptx) additionally need
# LibreOffice installed; MinerU raises a clear error if it is missing.
SUPPORTED_EXTENSIONS: list[str] = [".pdf", ".docx", ".pptx", ".doc", ".ppt"]


def _list_documents(folder: Path) -> list[Path]:
    """Return all supported documents under ``folder`` (recursive)."""
    files: list[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(folder.rglob(f"*{ext}"))
    return sorted(set(files))


async def ingest_folder(
    folder: str | Path = DOCUMENTS_DIR,
    max_workers: int = 2,
) -> None:
    """Ingest every supported document under ``folder`` into the knowledge graph.

    Documents that fail to parse are skipped and logged; the batch continues.

    Args:
        folder: Directory to scan recursively. Defaults to ./documents.
        max_workers: Parallel parse workers. Keep this modest (2) so concurrent
            Gemini calls stay under the free-tier rate limit.
    """
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")

    documents = _list_documents(folder_path)
    if not documents:
        logger.warning(
            "No supported documents (%s) found under %s. Add files and retry.",
            ", ".join(SUPPORTED_EXTENSIONS),
            folder_path,
        )
        return

    logger.info("Found %d document(s) under %s", len(documents), folder_path)
    rag = await get_rag()

    # process_folder_complete handles batching + stats. We call the per-file API
    # in a loop instead so a single failed document doesn't abort the run.
    succeeded, failed = 0, 0
    for doc in documents:
        try:
            logger.info("Ingesting %s ...", doc.name)
            await rag.process_document_complete(
                file_path=str(doc),
                output_dir=str(OUTPUT_DIR),
                display_stats=False,
            )
            succeeded += 1
        except Exception as exc:  # noqa: BLE001 - we intentionally skip + log
            failed += 1
            logger.error("Skipping %s — parse/ingest failed: %s", doc.name, exc)

    logger.info("Ingestion complete: %d succeeded, %d skipped", succeeded, failed)


async def ingest_folder_batch(
    folder: str | Path = DOCUMENTS_DIR,
    max_workers: int = 2,
) -> None:
    """Batch alternative using RAG-Anything's own folder processor + stats.

    This is faster for clean document sets but aborts if a file is unparseable.
    Prefer ``ingest_folder`` for mixed / untrusted folders.

    Args:
        folder: Directory to scan.
        max_workers: Parallel parse workers.
    """
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")

    rag = await get_rag()
    await rag.process_folder_complete(
        folder_path=str(folder_path),
        output_dir=str(OUTPUT_DIR),
        file_extensions=SUPPORTED_EXTENSIONS,
        recursive=True,
        max_workers=max_workers,
        display_stats=True,
    )
