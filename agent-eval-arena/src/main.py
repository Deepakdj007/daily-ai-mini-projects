"""Entrypoint: run every config through the harness and publish the leaderboard.

Inputs: none (reads .env and data/test_set.json via config.py and harness.py).
Outputs: terminal leaderboard, leaderboard.md, and traces in the Langfuse UI.
"""

from src.harness import run_all_configs
from src.leaderboard import publish_leaderboard


def main() -> None:
    """Run all configs against the test set and print + save the leaderboard."""
    print("Running agent eval arena across all configs...")
    experiment_results = run_all_configs()
    publish_leaderboard(experiment_results)


if __name__ == "__main__":
    main()
