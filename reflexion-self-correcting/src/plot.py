"""Plot the diminishing-returns curve: critic score against revision number.

The whole point of the Reflexion loop is to see *where* extra revisions stop
paying off. This draws the score per revision, marks the pass threshold, and
annotates the "knee" — the first revision whose gain over the previous one drops
below one point. That knee is the practical answer to "how many loops is enough".

Inputs:  scores (one int per revision) and an output path.
Outputs: a PNG written to disk; returns the knee revision (or None).
"""

from pathlib import Path

import matplotlib

# Use a non-interactive backend so this runs headless (no display needed).
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow backend selection)

from src.config import PASS_THRESHOLD  # noqa: E402


def find_knee(scores: list[int], min_gain: int = 1) -> int | None:
    """Return the first revision where the gain over the previous one is < min_gain.

    Revisions are 1-indexed to match how a reader counts drafts. Returns None if
    every step kept improving by at least min_gain (no flattening yet).
    """
    for i in range(1, len(scores)):
        if scores[i] - scores[i - 1] < min_gain:
            return i + 1  # 1-indexed revision number
    return None


def plot_curve(scores: list[int], out_path: Path) -> int | None:
    """Draw and save the score-vs-revision curve. Returns the knee revision."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    revisions = list(range(1, len(scores) + 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(revisions, scores, marker="o", color="#4f46e5", linewidth=2, label="critic score")
    ax.axhline(
        PASS_THRESHOLD,
        color="#16a34a",
        linestyle="--",
        linewidth=1.5,
        label=f"pass threshold ({PASS_THRESHOLD})",
    )

    knee = find_knee(scores)
    if knee is not None:
        ax.annotate(
            "diminishing returns",
            xy=(knee, scores[knee - 1]),
            xytext=(knee + 0.1, max(scores[knee - 1] - 2.5, 0.5)),
            arrowprops=dict(arrowstyle="->", color="#dc2626"),
            color="#dc2626",
            fontsize=10,
        )

    ax.set_title("Reflexion: quality vs. revision")
    ax.set_xlabel("revision number")
    ax.set_ylabel("critic score (0-10)")
    ax.set_xticks(revisions)
    ax.set_ylim(0, 10.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return knee
