"""CLI entry point: a small REPL around the offline agent.

Run:
  bash:        PYTHONPATH=. uv run python -m src.main
  PowerShell:  $env:PYTHONPATH="."; uv run python -m src.main
"""

from ollama import ResponseError

from src.agent import run_agent
from src.config import MODEL
from src.tools import TOOL_MAP


def main() -> None:
    """Print a banner, then loop: read a task, run the agent, print the answer."""
    print("=" * 60)
    print(f"Offline agent - model: {MODEL} (local, no API key)")
    print(f"Tools: {', '.join(TOOL_MAP)}")
    print("Type a task, or 'exit' to quit.")
    print("=" * 60)

    while True:
        task = input("\nyou> ").strip()
        if not task or task.lower() in {"exit", "quit"}:
            return
        try:
            answer = run_agent(task)
        except ConnectionError:
            print("Cannot reach Ollama. Start it with: ollama serve")
            continue
        except ResponseError as exc:
            print(f"Ollama error: {exc.error}")
            print(f"If the model is missing, run: ollama pull {MODEL}")
            continue
        print(f"\nagent> {answer}")


if __name__ == "__main__":
    main()
