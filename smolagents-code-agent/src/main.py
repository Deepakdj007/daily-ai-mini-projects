"""Interactive CLI for the smolagents code agent.

Type a problem, watch the agent write and run Python to solve it, see the final
answer, then ask another. Each problem starts with a fresh memory.

Run with:  PYTHONPATH=. uv run python -m src.main
"""

from src.agent import build_agent


def main() -> None:
    """Read problems in a loop and let the agent solve each one with code."""
    agent = build_agent()
    print("smolagents Code Agent (Groq + LiteLLM)")
    print("Describe any problem. The agent writes Python, runs it, and answers.")
    print("Press Enter on an empty line or type 'exit' to quit.\n")

    while True:
        task = input("Problem> ").strip()
        if not task or task.lower() == "exit":
            print("Bye.")
            return

        # reset=True (the default) clears memory so each problem is independent.
        answer = agent.run(task)
        print(f"\n=== Final answer ===\n{answer}\n")


if __name__ == "__main__":
    main()
