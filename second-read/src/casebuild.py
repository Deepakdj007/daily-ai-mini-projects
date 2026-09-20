"""Case folders on disk, and the one loader every command shares.

Inputs:  data/cases/<id>/case.json plus the files it lists, or any folder
Outputs: a Case (view + reference) with evidence hashed as the model will see it

`from_folder()` is what makes this a workstation rather than a benchmark: drop
a PNG, a `prior*.png` and a PDF into a directory and every command works. The
reference is simply empty, so the grading columns go blank and nothing else
changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src import config, images
from src.types import Case, CaseView, Evidence

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

# Public demo assets, none of them gated and none behind a login. The two
# longitudinal films are Google's own MedGemma notebook examples; the CXR is
# the public-domain Wikimedia film their localization notebook uses; the CT is
# a downsampled abdominal volume from the niivue demo set.
DEMO_ASSETS: dict[str, str] = {
    "cxr_wikimedia.png":
        "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png",
    "longitudinal_before.png":
        "https://storage.googleapis.com/hai-cd3-foundations-public-vault-entry/med_gemma/"
        "colab_example/cxr/longitudinal_cxr_before.png",
    "longitudinal_after.png":
        "https://storage.googleapis.com/hai-cd3-foundations-public-vault-entry/med_gemma/"
        "colab_example/cxr/longitudinal_cxr_after.png",
    "CT_Abdo.nii.gz":
        "https://raw.githubusercontent.com/niivue/niivue-demo-images/main/CT_Abdo.nii.gz",
}
# Wikimedia rejects a default urllib agent with HTTP 429, so identify the tool.
USER_AGENT = "second-read/0.1 (datasciencebrain tutorial project)"
# How a dropped-in file says "this is the earlier study". Checked against the
# stem, so `longitudinal_before.png` and `2024-prior-frontal.jpg` both land.
PRIOR_WORDS = ("prior", "before", "previous", "baseline", "old")


def evidence_bytes(item: Evidence) -> bytes:
    """The PNG the model is shown for one evidence item - resized, or rasterised."""
    if item.modality == "lab":
        return images.pdf_page_png(item.path)
    # The prior is one pixel smaller than the current study on purpose: Ollama
    # can fuse two adjacent same-size images into one (ollama#17321).
    edge = config.IMAGE_EDGE - 1 if item.role == "prior" else config.IMAGE_EDGE
    return images.prepare(item.path, edge=edge)


def _evidence(eid: str, modality: str, path: Path, role: str) -> Evidence:
    """Build one item, hashing the bytes that will actually be sent."""
    probe = Evidence(eid, modality, path, "", role)  # type: ignore[arg-type]
    return Evidence(eid, modality, path, images.sha256(evidence_bytes(probe)), role)  # type: ignore[arg-type]


def save(case_id: str, *, files: dict[str, tuple[Path, str, str]], context: str = "",
         reference: dict[str, Any] | None = None) -> Path:
    """Write case.json next to the evidence files. `files` maps eid -> (path, modality, role)."""
    folder = config.CASES_DIR / case_id
    folder.mkdir(parents=True, exist_ok=True)
    record = {
        "case_id": case_id, "context": context,
        "evidence": [{"eid": eid, "file": path.name, "modality": modality, "role": role}
                     for eid, (path, modality, role) in files.items()],
        "reference": reference or {},
    }
    (folder / "case.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return folder


def load(case_id: str) -> Case:
    """A case from data/cases, or from any folder path."""
    folder = config.CASES_DIR / case_id
    if not (folder / "case.json").exists():
        candidate = Path(case_id)
        if candidate.is_dir():
            return from_folder(candidate)
        raise SystemExit(f"no case {case_id!r} under {config.CASES_DIR} and no such folder")
    record = json.loads((folder / "case.json").read_text(encoding="utf-8"))
    evidence = tuple(_evidence(e["eid"], e["modality"], folder / e["file"], e["role"])
                     for e in record["evidence"])
    return Case(CaseView(record["case_id"], evidence, record.get("context", "")), record.get("reference", {}))


def from_folder(folder: Path) -> Case:
    """Infer a case from whatever files are in a directory. No reference, by definition."""
    items: list[Evidence] = []
    for path in sorted(folder.iterdir()):
        suffix = path.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            stem = path.stem.lower()
            role = "prior" if any(word in stem for word in PRIOR_WORDS) else "current"
            items.append(_evidence(path.stem, "cxr", path, role))
        elif suffix == ".pdf":
            items.append(_evidence(path.stem, "lab", path, "support"))
    if not items:
        raise SystemExit(f"{folder} holds no PNG/JPG or PDF files")
    context_file = folder / "context.txt"
    context = context_file.read_text(encoding="utf-8").strip() if context_file.exists() else ""
    return Case(CaseView(folder.name, tuple(items), context), {})


def list_cases(prefix: str = "") -> list[str]:
    """Case ids on disk, optionally filtered by prefix (CXR-, LAB-)."""
    return sorted(p.name for p in config.CASES_DIR.iterdir()
                  if (p / "case.json").exists() and p.name.startswith(prefix))


def fetch_demo_assets() -> dict[str, str]:
    """Download the four public demo files that are not on disk. Returns name -> status."""
    import urllib.error
    import urllib.request

    status: dict[str, str] = {}
    for name, url in DEMO_ASSETS.items():
        target = config.DEMO_DIR / name
        if target.exists() and target.stat().st_size > 0:
            status[name] = "present"
            continue
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=120) as response:
                target.write_bytes(response.read())
            status[name] = f"{target.stat().st_size:,} bytes"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            status[name] = f"FAILED ({exc})"
    return status


def build_pair_case() -> str | None:
    """The one longitudinal case: Google's two public demo films, prior and current."""
    before = config.DEMO_DIR / "longitudinal_before.png"
    after = config.DEMO_DIR / "longitudinal_after.png"
    if not (before.exists() and after.exists()):
        return None
    folder = config.CASES_DIR / "PAIR-demo"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "prior.png").write_bytes(before.read_bytes())
    (folder / "frontal.png").write_bytes(after.read_bytes())
    save("PAIR-demo",
         files={"prior": (folder / "prior.png", "cxr", "prior"),
                "frontal": (folder / "frontal.png", "cxr", "current")},
         context="Follow-up chest radiograph.")
    return "PAIR-demo"


if __name__ == "__main__":
    demo = config.DEMO_DIR
    if (demo / "longitudinal_before.png").exists():
        case = from_folder(demo)
        print(f"demo folder -> {len(case.view.evidence)} evidence items:")
        for item in case.view.evidence:
            print(f"  {item.eid:24s} {item.modality:4s} {item.role:8s} {item.sha256[:19]}...")
        assert case.reference == {}
        assert case.view.by_role("prior"), "the before/after pair must yield a prior"
        print(f"built {build_pair_case()}")
    print("OK - casebuild")
