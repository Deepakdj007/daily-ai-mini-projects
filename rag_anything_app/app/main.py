"""CLI entry point: `ingest` a folder or `chat` with the documents.

Usage (always with the mandatory PYTHONPATH prefix):

    PYTHONPATH=. uv run python -m app.main ingest ./documents
    PYTHONPATH=. uv run python -m app.main chat
    PYTHONPATH=. uv run python -m app.main ui
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from .chat import ask
from .config import DOCUMENTS_DIR, finalize
from .ingest import ingest_folder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rag_anything_app")


async def _run_ingest(folder: str, max_workers: int) -> None:
    """Ingest a folder of documents into the knowledge graph."""
    try:
        await ingest_folder(folder, max_workers=max_workers)
    finally:
        await finalize()


async def _run_chat() -> None:
    """Interactive terminal chat loop over the ingested documents."""
    print("Chat ready. Type a question, or 'exit' to quit.\n")
    loop = asyncio.get_event_loop()
    try:
        while True:
            question = (await loop.run_in_executor(None, input, "you> ")).strip()
            if question.lower() in {"exit", "quit"}:
                break
            if not question:
                continue
            try:
                answer = await ask(question)
            except Exception as exc:  # noqa: BLE001
                print(f"[error] {exc}\n")
                continue
            print(f"\nrag> {answer}\n")
    finally:
        await finalize()


def main() -> None:
    """Parse args and dispatch to the requested subcommand."""
    parser = argparse.ArgumentParser(description="RAG-Anything document chat")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Parse a folder into the knowledge graph")
    p_ingest.add_argument("folder", nargs="?", default=str(DOCUMENTS_DIR))
    p_ingest.add_argument("--max-workers", type=int, default=2)

    sub.add_parser("chat", help="Ask questions in the terminal")
    sub.add_parser("ui", help="Launch the Gradio web UI")

    args = parser.parse_args()

    if args.command == "ingest":
        asyncio.run(_run_ingest(args.folder, args.max_workers))
    elif args.command == "chat":
        asyncio.run(_run_chat())
    elif args.command == "ui":
        from .ui import launch  # lazy import so CLI users don't need gradio loaded

        launch()


if __name__ == "__main__":
    main()
