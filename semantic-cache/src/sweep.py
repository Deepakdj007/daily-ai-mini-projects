"""Pick a similarity threshold from data instead of guessing one.

Inputs: TRAFFIC (questions tagged with the intent they really ask about).
Outputs: for each threshold, how many calls the cache would skip and how many
of those skips would hand back an answer to a different question (a false hit).
Costs nothing: it embeds locally and replays the cache logic in memory.
"""

from dataclasses import dataclass

import numpy as np

from src.cache import EMBEDDER
from src.traffic import TRAFFIC


@dataclass
class SweepPoint:
    """The result of replaying the traffic at one threshold."""

    threshold: float
    hits: int
    false_hits: int
    example: str = ""  # one false hit, written out, if there was one


def similarity_matrix(questions: list[str]) -> np.ndarray:
    """Embed every question with the same model the cache uses; return cosine sims."""
    # One at a time: without padding, questions of different lengths can't share a batch.
    vectors = np.array([EMBEDDER([question])[0] for question in questions])
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors @ vectors.T


def replay(sims: np.ndarray, threshold: float) -> SweepPoint:
    """Run the traffic through an empty cache: a miss is stored, a hit is graded."""
    intents = [intent for _, intent in TRAFFIC]
    cached: list[int] = []
    point = SweepPoint(threshold, 0, 0)
    for i, (question, intent) in enumerate(TRAFFIC):
        best = max(cached, key=lambda j: sims[i, j], default=None)
        if best is None or sims[i, best] < threshold:
            cached.append(i)
            continue
        point.hits += 1
        if intents[best] != intent:
            point.false_hits += 1
            point.example = point.example or (
                f'"{question}" got the answer to "{TRAFFIC[best][0]}" ({sims[i, best]:.2f})'
            )
    return point


def closest_wrong_pair(sims: np.ndarray) -> tuple[float, str, str]:
    """Return the most similar pair of questions that ask different things."""
    best = (0.0, "", "")
    for i, (qi, ii) in enumerate(TRAFFIC):
        for j, (qj, ij) in enumerate(TRAFFIC[:i]):
            if ii != ij and sims[i, j] > best[0]:
                best = (float(sims[i, j]), qi, qj)
    return best


def sweep(thresholds: list[float]) -> tuple[list[SweepPoint], tuple[float, str, str]]:
    """Replay the traffic at every threshold; also return the closest wrong pair."""
    sims = similarity_matrix([question for question, _ in TRAFFIC])
    return [replay(sims, t) for t in thresholds], closest_wrong_pair(sims)
