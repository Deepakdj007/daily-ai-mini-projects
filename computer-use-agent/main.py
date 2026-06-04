"""Entry point: give the agent a task and run the See -> Think -> Act loop.

Run: PYTHONPATH=. uv run python main.py "open the start menu"

Safety: move your mouse to a screen corner at any time to abort (failsafe).
"""

import sys

from dotenv import load_dotenv

from src.config import MAX_STEPS
from src.graph import build_graph

load_dotenv()


def main() -> None:
    task = " ".join(sys.argv[1:]).strip() or input("Task: ").strip()
    print(f"Task: {task}")
    print("Failsafe: slam the mouse into a screen corner to abort.\n")

    agent = build_graph()
    final = agent.invoke(
        {
            "task": task,
            "screenshot": None,
            "action": None,
            "step": 0,
            "done": False,
            "history": [],
        },
        config={"recursion_limit": MAX_STEPS * 3 + 10},
    )

    print("\nFinished." if final["done"] else "\nStopped at step cap.")
    print(f"Steps taken: {final['step']}")
    for i, entry in enumerate(final["history"], 1):
        print(f"  {i}. {entry}")


if __name__ == "__main__":
    main()
