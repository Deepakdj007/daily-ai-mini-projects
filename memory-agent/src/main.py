# src/main.py
# ─────────────────────────────────────────────────────────────
# Entry point for the persistent memory agent.
# Run with: PYTHONPATH=. uv run python src/main.py
# ─────────────────────────────────────────────────────────────

from src.memory import create_memory
from src.agent import run_agent


def main() -> None:
    """
    Initialize mem0 and start the chat agent.

    Prompts for a user_id so memories are isolated per user.
    The same user_id across sessions = the agent remembers you.
    A different user_id = fresh memory space.
    """
    print("Initializing memory engine...")
    print("(First run: downloads the embedding model ~90MB)\n")

    # Create the Memory instance — this connects to Qdrant and loads
    # the HuggingFace embedding model into memory.
    memory = create_memory()

    # The user_id determines whose memories we're working with.
    # Using the same id across sessions = persistent memory.
    user_id = input("Enter your name (this is your memory ID): ").strip()

    if not user_id:
        user_id = "default_user"
        print(f"No name entered. Using '{user_id}'.")

    run_agent(memory, user_id)


if __name__ == "__main__":
    main()