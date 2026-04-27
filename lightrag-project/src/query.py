import asyncio
import sys
from src.rag_engine import build_rag
from lightrag import QueryParam

VALID_MODES = ["naive", "local", "global", "hybrid", "mix"]

async def run_query(question: str, mode: str = "mix") -> None:
    if mode not in VALID_MODES:
        print(f"Invalid mode '{mode}'. Choose from: {', '.join(VALID_MODES)}")
        sys.exit(1)

    rag = build_rag()
    await rag.initialize_storages()

    print(f"\n[Mode: {mode}] {question}\n{'─' * 60}")

    result = await rag.aquery(
        question,
        param=QueryParam(mode=mode),
    )

    print(result)
    await rag.finalize_storages()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run src/query.py 'your question' [mode]")
        print(f"Modes: {', '.join(VALID_MODES)} (default: mix)")
        sys.exit(1)

    question = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "mix"
    asyncio.run(run_query(question, mode))