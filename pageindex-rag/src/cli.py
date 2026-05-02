import sys
from src.ingest import ingest, load_doc_id
from src.query import DocumentChat

def run_cli():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  uv run python -m src.cli ingest <path/to/file.pdf>")
        print("  uv run python -m src.cli chat <filename-stem>")
        print("  uv run python -m src.cli trace <filename-stem>")
        sys.exit(1)

    command = sys.argv[1]
    arg = sys.argv[2]

    if command == "ingest":
        _cmd_ingest(arg)
    elif command == "chat":
        _cmd_chat(arg)
    elif command == "trace":
        _cmd_trace(arg)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

def _cmd_ingest(pdf_path: str):
    tree = ingest(pdf_path)
    top_sections = tree.get("result", [])
    print(f"\nDoc ID: {tree['doc_id']}")
    print(f"Top-level sections ({len(top_sections)}):")
    for section in top_sections:
        print(f"  - {section['title']} (page {section.get('page_index', '?')})")

def _cmd_chat(filename_stem: str):
    doc_id = load_doc_id(filename_stem)
    print(f"Chatting with: {filename_stem} (doc_id: {doc_id})")
    print("Type your question. 'reset' clears history. 'quit' exits.\n")

    chat = DocumentChat(doc_id)

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() == "quit":
            break
        if question.lower() == "reset":
            chat.reset()
            print("History cleared.\n")
            continue

        print("\nPageIndex:", end=" ", flush=True)
        print(chat.ask(question))
        print()

def _cmd_trace(filename_stem: str):
    doc_id = load_doc_id(filename_stem)
    print(f"Trace mode: {filename_stem} (doc_id: {doc_id})")
    print("You will see which sections PageIndex visits before answering.\n")

    chat = DocumentChat(doc_id)

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() == "quit":
            break
        if question.lower() == "reset":
            chat.reset()
            print("History cleared.\n")
            continue

        print()
        for event in chat.ask_stream_with_trace(question):
            if event["type"] == "trace":
                print(f"  [{event['content']}]", flush=True)
            else:
                print(event["content"], end="", flush=True)
        print("\n")


if __name__ == "__main__":
    run_cli()