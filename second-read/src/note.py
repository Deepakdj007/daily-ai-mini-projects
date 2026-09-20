"""The case note: write it, lift its claims, and render the audited version.

Inputs:  a CaseView and a model; later, the audits on each claim
Outputs: the note text, a list of Claims, and output/notes/<case>.md

Claims are lifted by the vocabulary scanner, not by asking the model to split
its own prose. A model splitting its own note would decide which of its
sentences count, and that is the one decision the audit must not delegate.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

from src import claims, config, llm, prompts
from src.claims import FINDINGS
from src.types import Audit, CaseView, Claim, Evidence

FLAG_LABEL = {
    "grounded": "evidence-backed",
    "image_shaped": "image-shaped: said this on another patient's film too",
    "prior": "LANGUAGE PRIOR: said this with the image removed",
    "unstable": "unstable: the answer did not reproduce",
    "swap_uninformative": "not swap-tested",
}


def current_image(view: CaseView) -> Evidence:
    """The film the note is about: the first current-role chest image."""
    for item in view.evidence:
        if item.modality == "cxr" and item.role == "current":
            return item
    raise SystemExit(f"{view.case_id} has no current chest image")


def write(model: str, view: CaseView, png: bytes) -> tuple[str, llm.Completion]:
    """The findings/impression note for one film."""
    out = llm.complete(model, prompts.note(view), (png,), num_predict=config.NUM_PREDICT_NOTE)
    return out.text, out


def sections(text: str) -> dict[str, str]:
    """Split on the two headings the prompt asked for."""
    findings = re.search(r"FINDINGS:?\s*(.*?)(?=IMPRESSION:|\Z)", text, re.I | re.S)
    impression = re.search(r"IMPRESSION:?\s*(.*)", text, re.I | re.S)
    return {"findings": (findings.group(1).strip() if findings else text.strip()),
            "impression": (impression.group(1).strip() if impression else "")}


def lift(text: str, view: CaseView) -> list[Claim]:
    """One Claim per vocabulary finding the note took a yes/no position on."""
    eid = current_image(view).eid
    out: list[Claim] = []
    for number, (finding, mention) in enumerate(claims.resolve(claims.scan(text)).items(), start=1):
        if mention.polarity in ("yes", "no"):
            out.append(Claim(f"{view.case_id}-c{number:02d}", finding, mention.polarity,
                             mention.sentence, (eid,)))
    return out


def render(view: CaseView, text: str, audits: Sequence[Audit], extras: Sequence[str] = ()) -> Path:
    """The audited note as markdown: the prose, then every claim with its tag."""
    parts = sections(text)
    lines = [f"# {view.case_id} - second read", "",
             "> Every claim below was re-asked with the image removed and with another patient's "
             "image in its place. A claim the model still makes without the image is marked as "
             "language prior, not evidence. Nothing here is a clinical opinion.", "",
             "## Findings", "", parts["findings"], "", "## Impression", "", parts["impression"] or "-", "",
             "## Claim audit", "",
             "Each cell is the model's own answer to *is this finding present?* under one condition. "
             "The only thing that changes between the columns is which image travelled with the question.",
             "",
             "| claim | finding | in the note | asked again, with the film | with no film | "
             "with a stranger's film | verdict |",
             "|---|---|:--:|:--:|:--:|:--:|---|"]
    for audit in audits:
        answers = {p.condition: p.polarity for p in audit.probes}
        lines.append(f"| {audit.cid} | {FINDINGS[audit.finding].phrase} | {audit.asserted} | "
                     f"{answers.get('sighted', '-')} | {answers.get('blind', '-')} | "
                     f"{answers.get('shuffled', '-')} | {FLAG_LABEL[audit.verdict]} |")
    flagged = sum(a.verdict == "prior" for a in audits)
    lines += ["", f"**{flagged} of {len(audits)} claims came back unchanged when the film was deleted.** "
              "Those are language prior, not evidence.", ""]
    for extra in extras:
        lines += [extra, ""]
    path = config.NOTES_DIR / f"{view.case_id}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    sample = ("FINDINGS: The heart size is normal. The lungs are clear. There is no pleural effusion "
              "or pneumothorax. Mild degenerative changes of the spine.\nIMPRESSION: No acute disease.")
    view = CaseView("demo", (Evidence("frontal", "cxr", Path("x.png"), "sha256:0"),), "")
    got = {c.finding: c.polarity for c in lift(sample, view)}
    assert got == {"cardiomegaly": "no", "consolidation": "no", "pleural_effusion": "no",
                   "pneumothorax": "no", "degenerative": "yes"}, got
    assert sections(sample)["impression"] == "No acute disease."

    from src.types import Audit, ProbeAnswer
    probe = lambda c, p: ProbeAnswer(c, p, "", (), None, 0, 0.1, 0, False)  # noqa: E731
    audits = [Audit("demo-c01", "cardiomegaly", "yes",
                    (probe("sighted", "yes"), probe("blind", "yes"), probe("shuffled", "no")),
                    "prior", "said it blind", reference="yes")]
    out = render(view, sample, audits)
    body = out.read_text(encoding="utf-8")
    assert body.count("|") > 12 and "language prior" in body.lower(), body
    print(f"OK - note, rendered {out.name}")
