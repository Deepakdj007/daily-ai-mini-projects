"""Synthetic lab reports: a PDF and its exact ground truth, minted in one pass.

Inputs:  a report count and a seed
Outputs: data/cases/LAB-nn/labs.pdf and data/reference/lab_truth.json

The values are the point. A ferritin of 187.4 has an answer space in the
thousands and appears exactly once on its page, so a model that produces it
with the page removed has seen the page. That gives the audit machinery a floor
that is zero by construction - the one check a chest film can never provide.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import inch
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src import config

# (panel, method, specimen, tests). A test is (name, unit, low, high, decimals).
# Decimals are chosen so every printed value carries three or four significant
# digits: "4.27" is unguessable in a way "4.2" is not.
PANELS = (
    ("Complete Blood Count", "Flow cytometry", "Whole blood (EDTA)", (
        ("Hemoglobin", "g/dL", 12.0, 17.5, 2), ("White Blood Cell Count", "x10^3/uL", 4.0, 11.0, 2),
        ("Platelet Count", "x10^3/uL", 150, 410, 0), ("Hematocrit", "%", 36.0, 50.0, 1),
        ("Mean Corpuscular Volume", "fL", 80.0, 100.0, 1))),
    ("Basic Metabolic Panel", "Ion-selective electrode", "Serum", (
        ("Sodium", "mmol/L", 136, 145, 1), ("Potassium", "mmol/L", 3.5, 5.1, 2),
        ("Creatinine", "mg/dL", 0.6, 1.3, 2), ("Glucose (fasting)", "mg/dL", 70, 99, 1),
        ("Blood Urea Nitrogen", "mg/dL", 7, 20, 1))),
    ("Liver Function", "Spectrophotometry", "Serum", (
        ("Alanine Aminotransferase", "U/L", 7, 56, 1), ("Aspartate Aminotransferase", "U/L", 10, 40, 1),
        ("Total Bilirubin", "mg/dL", 0.1, 1.2, 2), ("Albumin", "g/dL", 3.5, 5.0, 2))),
    ("Lipid Profile", "Enzymatic colorimetric", "Serum", (
        ("Total Cholesterol", "mg/dL", 125, 200, 1), ("LDL Cholesterol", "mg/dL", 50, 130, 1),
        ("HDL Cholesterol", "mg/dL", 40, 90, 1), ("Triglycerides", "mg/dL", 50, 150, 1))),
    ("Endocrine and Iron", "Chemiluminescent immunoassay", "Serum", (
        ("Thyroid Stimulating Hormone", "uIU/mL", 0.40, 4.00, 3), ("Ferritin", "ng/mL", 30, 400, 1),
        ("HbA1c", "%", 4.0, 5.6, 2), ("Vitamin D (25-OH)", "ng/mL", 30, 100, 1))),
)
PAGE = (6 * inch, 6 * inch)  # square: 900x900 px at 150 dpi, so the 896 squash does not stretch it
FIELDS = ("name", "result", "unit", "range", "panel", "method", "specimen", "sample_collection_time")


def _value(rng: random.Random, low: float, high: float, decimals: int) -> float:
    """Mostly in range; one in five lands 5-40% outside it so flags appear.

    Out-of-range values scale with the boundary, not the span: a ferritin range
    of 30-400 would otherwise produce a "low" of 5, which is a two-digit value
    and no longer unguessable.
    """
    if rng.random() < 0.2:
        value = high * rng.uniform(1.05, 1.4) if rng.random() < 0.5 else low * rng.uniform(0.6, 0.95)
    else:
        value = rng.uniform(low, high)
    return round(value, decimals)


def _print(value: float, decimals: int) -> str:
    """At least three significant digits on the page: a BUN of 5.9 becomes 5.93."""
    digits = len(str(int(value))) if value >= 1 else 0
    return f"{value:.{max(decimals, 3 - digits)}f}"


def mint(case_id: str, seed: int) -> dict:
    """One report's ground truth: unique result strings, the paper's eight fields."""
    rng = random.Random(seed)
    collected = datetime(2026, 1, 1) + timedelta(days=rng.randint(0, 240), hours=rng.randint(7, 11),
                                                 minutes=rng.randint(0, 59), seconds=rng.randint(0, 59))
    stamp = collected.strftime("%d-%m-%Y %H:%M:%S")
    tests: list[dict] = []
    seen: set[str] = set()
    for panel, method, specimen in rng.sample([(p, m, s) for p, m, s, _ in PANELS], 3):
        rows = next(t for p, _, _, t in PANELS if p == panel)
        for name, unit, low, high, decimals in rows:
            result = _print(_value(rng, low, high, decimals), decimals)
            while result in seen:
                result = _print(_value(rng, low, high, decimals), decimals)
            seen.add(result)
            tests.append({"name": name, "result": result, "unit": unit,
                          "range": f"{low:g}-{high:g}", "panel": panel, "method": method,
                          "specimen": specimen, "sample_collection_time": stamp,
                          "flag": "H" if float(result) > high else ("L" if float(result) < low else "")})
    return {"case_id": case_id, "patient_id": f"MD-{rng.randint(100000, 999999)}",
            "collected": stamp, "tests": tests}


def render(truth: dict, out_path: Path) -> Path:
    """Lay the report out as a ruled table with panel bands, the way a real one prints."""
    styles = getSampleStyleSheet()
    small = styles["BodyText"]
    small.fontSize, small.leading = 8, 10
    story = [Paragraph("<b>MERIDIAN DIAGNOSTICS</b> &nbsp; Clinical Laboratory Report", styles["Heading4"]),
             Paragraph(f"Patient ID: {truth['patient_id']} &nbsp;&nbsp; Sample collected: {truth['collected']}",
                       small), Spacer(1, 6)]
    data: list[list[str]] = [["Test", "Result", "Unit", "Reference", "Flag"]]
    style = [("FONTSIZE", (0, 0), (-1, -1), 8), ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
             ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe6ee")),
             ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
             ("TOPPADDING", (0, 0), (-1, -1), 2)]
    current = None
    for test in truth["tests"]:
        if test["panel"] != current:
            current = test["panel"]
            data.append([f"{current}  |  Method: {test['method']}  |  Specimen: {test['specimen']}", "", "", "", ""])
            row = len(data) - 1
            style += [("SPAN", (0, row), (-1, row)), ("BACKGROUND", (0, row), (-1, row), colors.HexColor("#f1f4f7")),
                      ("FONTNAME", (0, row), (-1, row), "Helvetica-Oblique")]
        data.append([test["name"], test["result"], test["unit"], test["range"], test["flag"]])
    table = Table(data, colWidths=[2.15 * inch, 0.8 * inch, 0.85 * inch, 0.95 * inch, 0.45 * inch])
    table.setStyle(TableStyle(style))
    story += [table, Spacer(1, 6),
              Paragraph("Results relate only to the sample tested. Reference intervals are adult ranges.", small)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(out_path), pagesize=PAGE, leftMargin=0.35 * inch, rightMargin=0.35 * inch,
                      topMargin=0.35 * inch, bottomMargin=0.35 * inch).build(story)
    return out_path


def build(n: int = config.N_LAB_REPORTS) -> dict[str, dict]:
    """Mint `n` reports, write the PDFs and the truth file, return the truth."""
    truth: dict[str, dict] = {}
    for index in range(1, n + 1):
        case_id = f"LAB-{index:02d}"
        truth[case_id] = mint(case_id, seed=1000 + index)
        render(truth[case_id], config.CASES_DIR / case_id / "labs.pdf")
    config.LAB_TRUTH_PATH.write_text(json.dumps(truth, indent=2), encoding="utf-8")
    return truth


def load_truth() -> dict[str, dict]:
    """The committed ground truth, keyed by case id."""
    if not config.LAB_TRUTH_PATH.exists():
        raise SystemExit("no lab truth yet - run `python -m src.main build` first")
    return json.loads(config.LAB_TRUTH_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    report = mint("LAB-00", seed=7)
    values = [t["result"] for t in report["tests"]]
    assert len(values) == len(set(values)), "result strings must be unique within a report"
    assert all(len(v.replace(".", "").lstrip("0")) >= 3 for v in values), values
    assert _print(5.9, 1) == "5.90" and _print(138.4, 1) == "138.4" and _print(0.31, 2) == "0.310"
    assert mint("LAB-00", seed=7) == report, "minting must be deterministic"
    path = render(report, config.OUTPUT_DIR / "selftest-lab.pdf")
    print(f"{len(report['tests'])} tests over {len({t['panel'] for t in report['tests']})} panels -> {path}")
    print(f"sample: {report['tests'][0]}")
    print("OK - labgen")
