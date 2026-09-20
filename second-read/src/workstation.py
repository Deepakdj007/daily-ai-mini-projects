"""The product: one case in, one audited record out.

Inputs:  a Case and a model
Outputs: output/cases/<id>.json, output/notes/<id>.md, overlay PNGs

This is what `case` runs and what the harness reuses. There is no separate
evaluation mode: the audit that scores a probe is the audit that tags a note.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date
from typing import Any

from src import audit, boxes, casebuild, claims, config, extract, llm, multi, note, probes
from src.claims import FINDINGS
from src.types import Audit, Case, Polarity


def manifest(model: str) -> dict[str, Any]:
    """What a record was produced with. Digest, not tag: tags move."""
    return {"date": date.today().isoformat(), "model": model, "digest": llm.digest(model),
            "prompt_version": config.PROMPT_VERSION, "temperature": config.TEMPERATURE,
            "top_k": config.TOP_K, "seeds": [config.SEED_MAIN, config.SEED_REPEAT],
            "num_ctx": config.NUM_CTX_SINGLE, "image_edge": config.IMAGE_EDGE}


def donor_pool() -> list[dict]:
    """The probe set doubles as the bank of reference-labelled films for swaps."""
    return probes.load()["probes"] if config.PROBES_PATH.exists() else []


def donor_png(finding: str, sighted: Polarity, case_id: str, pool: list[dict]) -> tuple[str, bytes] | None:
    """A film a radiologist labelled the opposite way, or None."""
    want: Polarity = "no" if sighted == "yes" else "yes"
    choice = audit.donor_for(finding, want, pool, case_id)
    if choice is None:
        return None
    return choice["case_id"], (config.CASES_DIR / choice["case_id"] / "frontal.png").read_bytes()


def reference_polarity(case: Case, finding: str) -> Polarity | None:
    """What the reference report says about one finding, if it says anything."""
    text = f"{case.reference.get('findings', '')} {case.reference.get('impression', '')}"
    return claims.polarity_of(text, finding) if text.strip() else None


def read_case(model: str, case: Case, pool: list[dict] | None = None) -> dict[str, Any]:
    """Note, claims, four probes per claim, verdicts; written to disk."""
    pool = donor_pool() if pool is None else pool
    view = case.view
    film = note.current_image(view)
    png = casebuild.evidence_bytes(film)
    text, wrote = note.write(model, view, png)
    lifted = note.lift(text, view)
    audits: list[Audit] = []
    for claim in lifted:
        donor = donor_png(claim.finding, claim.polarity, view.case_id, pool)
        answers = audit.run(model, FINDINGS[claim.finding].phrase, png, film.eid, donor)
        audits.append(audit.audit(claim.cid, claim.finding, claim.polarity, answers,
                                  reference_polarity(case, claim.finding)))
    record = {
        "manifest": manifest(model),
        "case": {"case_id": view.case_id, "context": view.context,
                 "evidence": [{**asdict(e), "path": str(e.path)} for e in view.evidence]},
        "note": {"text": text, "latency_s": wrote.latency_s, "prompt_tokens": wrote.prompt_tokens},
        "claims": [asdict(c) for c in lifted],
        "audits": [{**asdict(a), "flagged": a.flagged, "correct": a.correct} for a in audits],
        "summary": {"claims": len(audits), **{v: sum(a.verdict == v for a in audits)
                                            for v in ("grounded", "image_shaped", "prior", "unstable",
                                                      "swap_uninformative")}},
    }
    path = config.CASE_RECORDS_DIR / f"{view.case_id}.json"
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    record["note_path"] = str(note.render(view, text, audits))
    return record


def read_lab(model: str, case: Case) -> dict[str, Any]:
    """Sighted extraction scored exactly, and the blind floor that must be zero."""
    page = next(e for e in case.view.evidence if e.modality == "lab")
    png = casebuild.evidence_bytes(page)
    truth = case.reference.get("tests", [])
    tests, prose, sighted = extract.extract(model, png)
    blind_tests, blind_prose, blind = extract.extract(model, None)
    record = {
        "manifest": manifest(model), "case_id": case.view.case_id,
        "sighted": {"tests": tests, "score": extract.score(tests, truth) if truth else None,
                    "values_recovered": extract.values_recovered(tests, truth),
                    "latency_s": sighted.latency_s, "image_bytes_sent": sighted.image_bytes_sent},
        # values_recovered is the gate: a value under its own analyte name.
        # prose_collisions is the diagnostic it replaced - a bare number
        # anywhere in the answer, which the blind essay produces by chance.
        "blind": {"values_recovered": extract.values_recovered(blind_tests, truth),
                  "prose_collisions": extract.prose_collisions(blind_prose, truth),
                  "tests": blind_tests, "image_bytes_sent": blind.image_bytes_sent,
                  "answer": blind_prose[:600]},
    }
    (config.CASE_RECORDS_DIR / f"{case.view.case_id}.json").write_text(
        json.dumps(record, indent=2, default=str), encoding="utf-8")
    return record


def locate_case(model: str, case: Case, finding: str) -> dict[str, Any]:
    """Draw the box, then ask whether it is stable and whether it moves with the patient."""
    film = note.current_image(case.view)
    png = casebuild.evidence_bytes(film)
    phrase = boxes.LOCATE_NAMES.get(finding, finding)
    first, second, stable_iou = boxes.stability(model, png, phrase)
    # The stranger is a film a radiologist called negative for this finding,
    # never the same file: a box compared with itself proves nothing.
    stranger = donor_png(finding, "yes", case.view.case_id, donor_pool()) if finding in FINDINGS else None
    swap_iou, stranger_boxes = None, []
    if stranger is not None and stranger[1] != png:
        _, stranger_boxes, swap_iou = boxes.swap_sensitivity(model, png, stranger[1], phrase)
    overlay = boxes.overlay(png, first, case.view.case_id, finding) if first else None
    return {"asked_for": phrase, "boxes": [asdict(b) for b in first], "repeat": [asdict(b) for b in second],
            "stability_iou": stable_iou, "stranger": stranger[0] if stranger else None,
            "stranger_boxes": [asdict(b) for b in stranger_boxes], "swap_iou": swap_iou,
            "overlay": str(overlay) if overlay else None}


def compare_case(model: str, case: Case) -> dict[str, Any]:
    """Prior versus current, then the same with a stranger's prior."""
    from src import images
    current = casebuild.evidence_bytes(note.current_image(case.view))
    priors = case.view.by_role("prior")
    if not priors:
        raise SystemExit(f"{case.view.case_id} has no prior-role image")
    prior = casebuild.evidence_bytes(priors[0])
    stranger = images.prepare(config.DEMO_DIR / "cxr_wikimedia.png", edge=config.IMAGE_EDGE - 1)
    return multi.prior_swap(model, current, prior, stranger)


if __name__ == "__main__":
    print("workstation: read_case / read_lab / locate_case / compare_case")
