"""Interactive CLI for the self-healing code agent.

Ask the user for a coding task, run the agent, then replay every attempt so the
write -> fail -> fix loop is visible in the terminal before the final code.

Run with:  PYTHONPATH=. uv run python -m src.main
"""

from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.usage import UsageLimits

from src.agent import Deps, build_agent
from src.config import MAX_REQUESTS


def _print_attempts(deps: Deps) -> None:
    """Show each write-and-run cycle the agent went through."""
    for i, attempt in enumerate(deps.attempts, start=1):
        status = "PASSED" if attempt.ok else "FAILED"
        print(f"\n--- Attempt {i}: {status} ---")
        if not attempt.ok:
            # The traceback is the interesting part — it's what the agent read
            # and fixed on the next attempt.
            print(attempt.output)


def main() -> None:
    """Read a task, run the agent, and report the self-healing loop."""
    agent = build_agent()
    print("Self-Healing Code Agent (Groq + Pydantic AI)")
    print("Describe a coding task. The agent writes it, tests it, and fixes its")
    print("own bugs until the tests pass.\n")

    task = input("Task> ").strip()
    if not task:
        print("No task given. Exiting.")
        return

    deps = Deps()
    print("\nWorking...\n")

    try:
        result = agent.run_sync(
            task,
            deps=deps,
            usage_limits=UsageLimits(request_limit=MAX_REQUESTS),
        )
    except UsageLimitExceeded:
        _print_attempts(deps)
        print(
            f"\nGave up after {MAX_REQUESTS} requests without passing all tests. "
            "Try rephrasing the task or raising MAX_REQUESTS in src/config.py."
        )
        return

    _print_attempts(deps)
    print(f"\n=== Solved in {len(deps.attempts)} attempt(s) ===")
    print(f"\nSummary: {result.output.summary}\n")
    print("Final working code:\n")
    print(result.output.code)


if __name__ == "__main__":
    main()
