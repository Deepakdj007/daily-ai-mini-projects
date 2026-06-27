"""CLI entry point: run the Reflexion loop and save the email + the curve.

Streams each handoff so you can watch the score climb revision by revision, then
writes the final email to output/email.md, the adjudicator's verdict alongside
it, and the diminishing-returns curve to output/curve.png.

Run:  PYTHONPATH=. uv run python -m src.main
      PYTHONPATH=. uv run python -m src.main "your brief here"
"""

import sys
from pathlib import Path

from src.config import MAX_REVISIONS, require_keys
from src.graph import build_graph
from src.plot import plot_curve
from src.state import ReflexionState

# A deliberately under-specified brief so the first draft has room to improve and
# the loop visibly works. Override it by passing your own brief on the CLI.
DEFAULT_TASK = (
    "Reach out to the head of data at a mid-size Indian logistics company. "
    "We sell an ML tool that predicts late deliveries 48 hours early. "
    "Goal: book a 15-minute call."
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def run(task: str) -> None:
    """Execute the graph end to end and persist all artifacts."""
    require_keys()
    graph = build_graph()
    initial: ReflexionState = {"task": task, "revision": 0, "scores": [], "history": []}

    final: ReflexionState = {}
    print("Running Reflexion loop...\n")
    # stream_mode="values" yields the full state after each node, so we print the
    # handoff log as it grows.
    for state in graph.stream(initial, {"recursion_limit": 50}, stream_mode="values"):
        final = state
        if state.get("history"):
            print(f"  {state['history'][-1]}")

    scores = final.get("scores", [])
    print(f"\nFinished after {final.get('revision', 0)} revision(s).")
    print(f"Score trajectory: {scores}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    email_path = OUTPUT_DIR / "email.md"
    verdict_model = final.get("verdict_model", "adjudicator")
    email_path.write_text(
        f"# Final cold email (score {final.get('score', 0)}/10)\n\n"
        f"{final.get('draft', '')}\n\n"
        f"---\n\n## Adjudicator verdict ({verdict_model})\n\n"
        f"{final.get('final_verdict', '')}\n",
        encoding="utf-8",
    )

    curve_path = OUTPUT_DIR / "curve.png"
    knee = plot_curve(scores, curve_path) if scores else None

    print(f"\nEmail saved to:  {email_path}")
    print(f"Curve saved to:  {curve_path}")
    if knee is not None:
        print(f"Diminishing returns set in at revision {knee}.")
    elif len(scores) >= MAX_REVISIONS:
        print("Quality was still improving at the revision cap.")

    print(
        f"\n--- Adjudicator verdict ({final.get('verdict_model', 'n/a')}) ---\n"
        f"{final.get('final_verdict', '')}"
    )


def main() -> None:
    task = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TASK
    run(task)


if __name__ == "__main__":
    main()
