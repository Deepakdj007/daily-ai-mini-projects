# main.py
# ─────────────────────────────────────────────────────────────
# Entry point for the parallel news analyst.
# Configure your topics list here and run:
#   PYTHONPATH=. uv run python src/main.py
# The report is printed to the terminal and saved as report.md.
# ─────────────────────────────────────────────────────────────

import time
from graph import build_graph

# ── Configure your topics here ──────────────────────────────
# Change any of these to the topics you want covered today.
# You can add more — the parallel architecture handles N topics.
TOPICS = [
    "artificial intelligence breakthroughs",
    "Indian stock market",
    "climate change",
    "geopolitics Middle East",
    "open source software releases",
]
# ────────────────────────────────────────────────────────────

def main():
    """Run the parallel news analyst and save the report."""
    graph = build_graph()

    print(f"Launching {len(TOPICS)} parallel analyst agents...")
    print("Topics:", ", ".join(TOPICS))
    print()

    # Record start time to demonstrate the parallel speedup.
    start = time.perf_counter()

    # Invoke the graph with the topics list.
    # The graph handles all parallelism internally.
    result = graph.invoke({"topics": TOPICS})

    elapsed = time.perf_counter() - start

    # Print the report to the terminal.
    print(result["report"])
    print(f"\n⏱  Completed in {elapsed:.1f} seconds for {len(TOPICS)} topics")

    # Save the report to a file for easy sharing.
    with open("report.md", "w") as f:
        f.write(result["report"])
    print("✅ Report saved to report.md")


if __name__ == "__main__":
    main()