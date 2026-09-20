"""The audit: one claim, four conditions, one pure decision.

Inputs:  a finding phrase, the case image, a donor image, a model
Outputs: ProbeAnswers per condition and an Audit with its verdict

Every condition asks the byte-identical question. What differs is one thing:
`blind` sends no image, `shuffled` sends another patient's, `repeat` changes
the seed. `assert_single_switch()` checks that at import, because an audit
condition that quietly changed two things would credit one for the other's
work.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

from src import config, llm, prompts
from src.types import Audit, Condition, Polarity, ProbeAnswer, Verdict


@dataclass(frozen=True, slots=True)
class Arm:
    """One audit condition: what it changes and what was predicted before the run."""

    name: Condition
    images: str          # "case" | "none" | "donor"
    seed: int
    isolates: str
    prediction: str


LADDER: tuple[Arm, ...] = (
    Arm("sighted", "case", config.SEED_MAIN, "the product's own answer",
        "medgemma1.5 above 70% on balanced probes; the twin nearer 55%"),
    Arm("blind", "none", config.SEED_MAIN, "does the claim survive deleting the evidence",
        "at the 50% floor for both models, with medgemma1.5 agreeing with its sighted self more often"),
    Arm("shuffled", "donor", config.SEED_MAIN, "does it survive replacing the evidence",
        "between blind and sighted; the gap to sighted is what the specific film buys"),
    Arm("repeat", "case", config.SEED_REPEAT, "the noise floor at temperature zero",
        "under 5% disagreement - greedy decoding on a fixed prompt should be near-deterministic"),
)
BY_NAME = {arm.name: arm for arm in LADDER}


def assert_single_switch() -> None:
    """Every condition differs from `sighted` in exactly one field."""
    base = BY_NAME["sighted"]
    for arm in LADDER[1:]:
        changed = [f for f in ("images", "seed") if getattr(arm, f) != getattr(base, f)]
        if len(changed) != 1:
            raise AssertionError(f"{arm.name} changes {changed}; one switch per condition")


assert_single_switch()

_ABSTAIN = re.compile(r"(?:cannot|can't|unable to|no) (?:see|view|access)|no (?:image|radiograph) (?:is|was) "
                      r"(?:provided|attached|shown)|without (?:an|the) image|not provided", re.I)


def parse_polarity(text: str) -> Polarity:
    """The schema's final_answer; falling back to a Final Answer line, then an abstention."""
    payload = llm.parse_json(text)
    if isinstance(payload, dict) and payload.get("final_answer") in ("yes", "no"):
        return payload["final_answer"]  # type: ignore[return-value]
    answer = llm.final_answer(text)
    if answer is not None:
        head = re.match(r"[\W_]*(yes|no)\b", answer, re.I)
        if head:
            return head.group(1).lower()  # type: ignore[return-value]
    if _ABSTAIN.search(text):
        return "uncertain"
    tail = text.strip().splitlines()[-1] if text.strip() else ""
    has_yes, has_no = bool(re.search(r"\byes\b", tail, re.I)), bool(re.search(r"\bno\b", tail, re.I))
    if has_yes != has_no:
        return "yes" if has_yes else "no"
    return "unparsed"


def reasoning(text: str) -> str:
    """The schema's reasoning field, for the audit trail."""
    payload = llm.parse_json(text)
    return str(payload.get("reasoning", "")) if isinstance(payload, dict) else text


def ask(model: str, phrase: str, png: bytes | None, *, condition: Condition, seed: int,
        evidence_shown: Sequence[str], donor_case_id: str | None = None) -> ProbeAnswer:
    """One condition on one claim."""
    out = llm.complete(model, prompts.verify(phrase), (png,) if png else (), seed=seed,
                       num_predict=config.NUM_PREDICT_PROBE, schema=prompts.VERIFY_SCHEMA)
    return ProbeAnswer(condition, parse_polarity(out.text), out.text, tuple(evidence_shown),
                       donor_case_id, seed, out.latency_s, out.image_bytes_sent, out.truncated)


def run(model: str, phrase: str, png: bytes, eid: str, donor: tuple[str, bytes] | None,
        conditions: Sequence[Condition] = config.CONDITIONS) -> dict[Condition, ProbeAnswer]:
    """All requested conditions, sighted first so a stopped run still has the answer."""
    out: dict[Condition, ProbeAnswer] = {}
    for name in conditions:
        arm = BY_NAME[name]
        if arm.images == "case":
            out[name] = ask(model, phrase, png, condition=name, seed=arm.seed, evidence_shown=(eid,))
        elif arm.images == "none":
            out[name] = ask(model, phrase, None, condition=name, seed=arm.seed, evidence_shown=())
        elif donor is not None:
            out[name] = ask(model, phrase, donor[1], condition=name, seed=arm.seed,
                            evidence_shown=(f"donor:{donor[0]}",), donor_case_id=donor[0])
    return out


def decide(asserted: Polarity, probes: Mapping[str, ProbeAnswer]) -> tuple[Verdict, str]:
    """Pure. The order is the design: stability first, then prior, then the swap."""
    sighted = probes["sighted"]
    repeat = probes.get("repeat")
    if repeat is not None and repeat.polarity != sighted.polarity:
        return "unstable", f"the same call answered {sighted.polarity} then {repeat.polarity}"
    if sighted.polarity != asserted:
        return "unstable", f"the note said {asserted} but the targeted re-read said {sighted.polarity}"
    blind = probes.get("blind")
    if blind is not None and blind.polarity == sighted.polarity:
        return "prior", f"said {sighted.polarity} with no image in the prompt at all"
    shuffled = probes.get("shuffled")
    if shuffled is None:
        return "swap_uninformative", "no reference-opposite donor was available"
    if shuffled.polarity == sighted.polarity:
        return "image_shaped", f"said {sighted.polarity} on another patient's film too"
    return "grounded", "changed its answer when the image was removed and when it was swapped"


def audit(cid: str, finding: str, asserted: Polarity, probes: Mapping[str, ProbeAnswer],
          reference: Polarity | None = None) -> Audit:
    """Bundle the probes and the decision for one claim."""
    verdict, reason = decide(asserted, probes)
    ordered = tuple(probes[c] for c in config.CONDITIONS if c in probes)
    return Audit(cid, finding, asserted, ordered, verdict, reason, reference)


def donor_for(finding: str, want: Polarity, pool: Sequence[dict], exclude_case: str) -> dict | None:
    """A probe of the same finding whose reference polarity is `want`, chosen deterministically.

    For a sighted "yes" the donor must be a reference "no", and vice versa. A
    "yes" that survives on a film a radiologist called negative is the thing
    the swap exists to catch; on a positive film it would be the model being
    right.
    """
    options = [p for p in pool if p["finding"] == finding and p["answer"] == want
               and p["case_id"] != exclude_case]
    if not options:
        return None
    return random.Random(f"{exclude_case}|{finding}|{want}").choice(sorted(options, key=lambda p: p["pid"]))


if __name__ == "__main__":
    assert parse_polarity('{"reasoning": "the heart is large", "final_answer": "yes"}') == "yes"
    assert reasoning('{"reasoning": "the heart is large", "final_answer": "yes"}') == "the heart is large"
    assert parse_polarity("Reasoning...\nFinal Answer: yes") == "yes"
    assert parse_polarity("Final Answer: No.") == "no"
    assert parse_polarity("I cannot see any image attached to this request.") == "uncertain"
    assert parse_polarity("Something unrelated.") == "unparsed"

    def _p(cond, pol, seed=0):
        return ProbeAnswer(cond, pol, "", (), None, seed, 0.1, 0, False)

    both_yes = {"sighted": _p("sighted", "yes"), "blind": _p("blind", "yes"),
                "shuffled": _p("shuffled", "no"), "repeat": _p("repeat", "yes", 1)}
    assert decide("yes", both_yes)[0] == "prior"
    grounded = dict(both_yes, blind=_p("blind", "no"))
    assert decide("yes", grounded)[0] == "grounded"
    shaped = dict(grounded, shuffled=_p("shuffled", "yes"))
    assert decide("yes", shaped)[0] == "image_shaped"
    flappy = dict(grounded, repeat=_p("repeat", "no", 1))
    assert decide("yes", flappy)[0] == "unstable"
    assert decide("yes", {"sighted": _p("sighted", "yes"), "blind": _p("blind", "no")})[0] == "swap_uninformative"
    pool = [{"pid": "p1", "finding": "f", "answer": "no", "case_id": "A"},
            {"pid": "p2", "finding": "f", "answer": "no", "case_id": "B"}]
    assert donor_for("f", "no", pool, "A")["case_id"] == "B"
    assert donor_for("f", "yes", pool, "A") is None
    print("OK - audit: single switch holds, decision table, donor rule")
