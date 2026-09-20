"""The data model: evidence, cases, claims, probes, audits and boxes.

Inputs:  nothing - pure declarations
Outputs: frozen dataclasses shared by every module

One boundary here is load-bearing. `CaseView` is everything a prompt may see
and it has no reference field. `Case` wraps a view together with the reference
report used for grading. `prompts.py` imports only `CaseView`, so the reference
text has no type to travel in and cannot leak into a prompt by accident.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Modality = Literal["cxr", "lab", "ct", "text"]
Polarity = Literal["yes", "no", "uncertain", "unparsed"]
Condition = Literal["sighted", "blind", "shuffled", "repeat"]
Verdict = Literal["grounded", "image_shaped", "prior", "unstable", "swap_uninformative"]


@dataclass(frozen=True, slots=True)
class Evidence:
    """One file the model may be shown, with the hash of the bytes actually sent."""

    eid: str
    modality: Modality
    path: Path
    sha256: str
    role: Literal["current", "prior", "support"] = "current"


@dataclass(frozen=True, slots=True)
class CaseView:
    """Everything a prompt is allowed to see. Deliberately has no reference."""

    case_id: str
    evidence: tuple[Evidence, ...]
    context: str = ""

    def by_role(self, role: str) -> tuple[Evidence, ...]:
        """The evidence items carrying one role."""
        return tuple(item for item in self.evidence if item.role == role)


@dataclass(frozen=True, slots=True)
class Case:
    """A view plus the reference used for grading. Never handed to a prompt."""

    view: CaseView
    reference: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Mention:
    """One finding as a report or a note stated it."""

    finding: str
    polarity: Polarity
    sentence: str
    source: Literal["negation", "stated_normal", "assertion", "hedge"]


@dataclass(frozen=True, slots=True)
class Claim:
    """One atomic assertion lifted from a note, ready to be re-tested."""

    cid: str
    finding: str
    polarity: Polarity
    sentence: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProbeAnswer:
    """The model's answer to one claim under one condition."""

    condition: Condition
    polarity: Polarity
    raw: str
    evidence_shown: tuple[str, ...]
    donor_case_id: str | None
    seed: int
    latency_s: float
    image_bytes_sent: int
    truncated: bool


@dataclass(frozen=True, slots=True)
class Audit:
    """The four probes on one claim and what they add up to."""

    cid: str
    finding: str
    asserted: Polarity
    probes: tuple[ProbeAnswer, ...]
    verdict: Verdict
    reason: str
    reference: Polarity | None = None

    @property
    def flagged(self) -> bool:
        """The note's claim survives deleting the image: language prior."""
        return self.verdict == "prior"

    @property
    def correct(self) -> bool | None:
        """Agreement with the reference, when there is one. Orthogonal to `flagged`."""
        if self.reference is None or self.reference not in ("yes", "no"):
            return None
        return self.asserted == self.reference


@dataclass(frozen=True, slots=True)
class Box:
    """A bounding box in normalised [0, 1] coordinates, MedGemma's y-first order."""

    y0: float
    x0: float
    y1: float
    x1: float
    label: str

    def iou(self, other: "Box") -> float:
        """Intersection over union. Compares the model to itself - there is no gold box."""
        iy0, ix0 = max(self.y0, other.y0), max(self.x0, other.x0)
        iy1, ix1 = min(self.y1, other.y1), min(self.x1, other.x1)
        inter = max(0.0, iy1 - iy0) * max(0.0, ix1 - ix0)
        area = lambda b: max(0.0, b.y1 - b.y0) * max(0.0, b.x1 - b.x0)  # noqa: E731
        union = area(self) + area(other) - inter
        return inter / union if union > 0 else 0.0


if __name__ == "__main__":
    a = Box(0.2, 0.3, 0.6, 0.7, "heart")
    b = Box(0.3, 0.4, 0.7, 0.8, "heart")
    assert abs(a.iou(a) - 1.0) < 1e-9
    assert 0.0 < a.iou(b) < 1.0, a.iou(b)
    assert a.iou(Box(0.9, 0.9, 1.0, 1.0, "x")) == 0.0
    audit = Audit("c1", "cardiomegaly", "yes", (), "prior", "said it blind", reference="no")
    assert audit.flagged and audit.correct is False
    print(f"OK - iou {a.iou(b):.3f}, flagged={audit.flagged}, correct={audit.correct}")
