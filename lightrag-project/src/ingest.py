import asyncio
import sys
from pathlib import Path
from src.rag_engine import build_rag

async def ingest_file(path: str) -> None:
    file = Path(path)
    if not file.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    text = file.read_text(encoding="utf-8")
    print(f"Ingesting: {file.name} ({len(text):,} characters)")

    rag = build_rag()
    await rag.initialize_storages()

    await rag.ainsert(text, file_paths=str(file))

    await rag.finalize_storages()
    print("Done. Knowledge graph updated.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run src/ingest.py <path/to/file.txt>")
        sys.exit(1)
    asyncio.run(ingest_file(sys.argv[1]))