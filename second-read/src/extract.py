"""Lab-report extraction: page image in, the paper's eight fields out, scored exactly.

Inputs:  a rasterised report page (or none, for the blind floor) and a model
Outputs: a list of test dicts, the stage-one prose, and per-field scores

Two stages. The first shows the page with Google's own extraction prompt and
lets the model write freely. The second turns that prose into schema-valid
JSON with no image in the call. Forcing the schema on the first call would
suppress the reasoning and, on a table this dense, the numbers with it.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from src import config, llm, matcher, prompts
from src.labgen import FIELDS


class LabTest(BaseModel):
    """One row, in the vocabulary the model was evaluated against."""

    name: str
    result: str | None = None
    unit: str | None = None
    range: str | None = None
    panel: str | None = None
    method: str | None = None
    specimen: str | None = None
    sample_collection_time: str | None = None


class LabReport(BaseModel):
    """Ollama's grammar wants an object at the top, so the list is wrapped."""

    tests: list[LabTest]


def extract(model: str, page_png: bytes | None) -> tuple[list[dict[str, Any]], str, llm.Completion]:
    """Stage one with the image (if any), stage two text-only. Returns (tests, prose, stage1)."""
    stage1 = llm.complete(model, prompts.LAB_EXTRACT, (page_png,) if page_png else (),
                          num_predict=config.NUM_PREDICT_EXTRACT)
    parsed = llm.structure(model, stage1.text, LabReport.model_json_schema(),
                           instruction=prompts.LAB_STRUCTURE)
    tests = parsed.get("tests", []) if isinstance(parsed, dict) else []
    return [t for t in tests if isinstance(t, dict)], stage1.text, stage1


def _norm(value: Any) -> str:
    """Fold the spellings that are not disagreements: case, spaces, dash spacing, units."""
    text = matcher.normalize(str(value if value is not None else "")).lower()
    text = re.sub(r"\s*-\s*", "-", text)
    text = text.replace("μ", "u").replace("µ", "u").replace(" ", "")
    return text.rstrip(".")


def _row_for(name: str, predicted: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The predicted row for one gold test name, allowing either to abbreviate the other.

    Models shorten "Alanine Aminotransferase" to "ALT" and expand "HbA1c", so an
    exact name match is too strict. Containment either way is enough; an empty
    predicted name never matches, which a bare `in` test would let through.
    """
    for row in predicted:
        got = _norm(row.get("name"))
        if got and (got in name or name in got):
            return row
    return None


def score(predicted: list[dict[str, Any]], truth: list[dict[str, Any]]) -> dict[str, Any]:
    """Match rows by test name, then compare each of the eight fields exactly."""
    per_field = {field: [0, 0] for field in FIELDS}       # [correct, total]
    matched = 0
    for gold in truth:
        row = _row_for(_norm(gold["name"]), predicted)
        for field in FIELDS:
            per_field[field][1] += 1
            if row is not None and _norm(row.get(field)) == _norm(gold[field]):
                per_field[field][0] += 1
        matched += row is not None
    correct = sum(c for c, _ in per_field.values())
    total = sum(t for _, t in per_field.values())
    return {"rows_truth": len(truth), "rows_predicted": len(predicted), "rows_matched": matched,
            "fields_correct": correct, "fields_total": total,
            "field_accuracy": correct / total if total else 0.0,
            "per_field": {f: c / t if t else 0.0 for f, (c, t) in per_field.items()}}


def prose_collisions(text: str, truth: list[dict[str, Any]]) -> int:
    """How many minted result values appear anywhere in the model's output.

    This is a diagnostic, NOT a gate. Asked to extract a lab report with no
    page attached, MedGemma writes a six-thousand-character essay about how it
    would do the job, full of plausible example numbers. A one-decimal blood
    urea nitrogen inside a 7-20 reference range has roughly 130 possible
    values, so a loose "14.5" in that essay collides with a minted 14.5 often
    enough to be expected rather than alarming - it happened twice across 269
    values on the first full run. A bare substring scan cannot tell that from
    a leak, so the gate uses `values_recovered` instead.
    """
    return sum(matcher.matches(t["result"], text) for t in truth)


def values_recovered(predicted: list[dict[str, Any]], truth: list[dict[str, Any]]) -> int:
    """Minted values the model reports ATTACHED TO THE RIGHT TEST NAME.

    Blind, this must be zero. Guessing that some lab report somewhere contains
    the number 14.5 is free; naming the analyte it belongs to on a page you
    were never shown is not. Pairing the value with the name is what turns a
    coincidence into evidence that the page arrived.
    """
    rows = [row for row in predicted if isinstance(row, dict)]
    hits = 0
    for test in truth:
        row = _row_for(_norm(test["name"]), rows)
        if row is not None and matcher.matches(test["result"], str(row.get("result", ""))):
            hits += 1
    return hits


if __name__ == "__main__":
    truth = [{"name": "Sodium", "result": "138.4", "unit": "mmol/L", "range": "136-145",
              "panel": "Basic Metabolic Panel", "method": "Ion-selective electrode",
              "specimen": "Serum", "sample_collection_time": "03-04-2026 08:15:22"}]
    pred = [{"name": "sodium", "result": "138.4", "unit": "mmol / L", "range": "136 - 145",
             "panel": "Basic Metabolic Panel", "method": "Ion-selective electrode",
             "specimen": "serum", "sample_collection_time": "03-04-2026 08:15:22"}]
    got = score(pred, truth)
    assert got["fields_correct"] == 8, got
    assert score([], truth)["fields_correct"] == 0

    assert values_recovered(pred, truth) == 1, "right name, right value"
    assert values_recovered([{"name": "Potassium", "result": "138.4"}], truth) == 0, \
        "the value alone is a coincidence; it has to arrive under its own name"
    assert values_recovered([{"name": "Sodium", "result": "1138.4"}], truth) == 0, \
        "boundary matching, not substring"
    assert values_recovered([], truth) == 0
    assert values_recovered([{"name": "", "result": "138.4"}], truth) == 0, "an empty name matches nothing"
    assert _row_for("sodium", [{"name": "Serum Sodium"}])["name"] == "Serum Sodium", "abbreviation either way"

    # The failure that made this function necessary: a loose number in prose.
    assert prose_collisions("a typical sodium is 138.4", truth) == 1
    assert prose_collisions("nothing numeric here", truth) == 0
    print(LabReport.model_json_schema()["properties"]["tests"]["type"])
    print("OK - extract")
