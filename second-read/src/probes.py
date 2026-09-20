"""Authoring the probe set from IU-Xray: balanced per finding, one per study, hand-checked.

Inputs:  the open-i parquet shards on the Hugging Face Hub
Outputs: data/reference/openi_index.json, data/reference/probes.json, data/cases/CXR-*/

Three rules make the blind floor an identity instead of an estimate. A "no"
comes only from an explicit negation or an explicit statement of normal, never
from silence. Each finding contributes exactly as many yes as no probes, with
the same question text on both sides. Each study appears in at most one probe.
A blind model that guesses any fixed answer per finding therefore scores
exactly 50%, and a blind score away from 50% is a leak, not a result.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from src import casebuild, claims, config, images
from src.claims import FINDINGS

DATASET_GLOB = "datasets/ykumards/open-i/data/*.parquet"
TEXT_COLUMNS = ("uid", "MeSH", "Problems", "image", "indication", "comparison", "findings", "impression")
RNG_SEED = 20260912
SHADOW_PER_FINDING = 5

# The dataset's own manual MeSH index, used as a second opinion on every label.
# A "yes" whose MeSH never mentions the finding, or a "no" whose MeSH does,
# is a report contradicting itself - "No nodules or masses" on a study indexed
# under Granuloma - and neither side of that is a probe.
MESH_KEYS = {
    "cardiomegaly": r"Cardiomegaly|Heart/enlarged|Cardiac.*enlarg",
    "pleural_effusion": r"Effusion|Pleural Fluid",
    "consolidation": r"Opacity|Consolidation|Airspace|Infiltrat|Pneumonia",
    "pulmonary_edema": r"Edema|Congestion",
    "pneumothorax": r"Pneumothorax",
    "nodule": r"Nodule|Mass|Granuloma|Density.*round|Calcinosis/lung",
    "atelectasis": r"Atelectasis",
    "hyperinflation": r"hyperdistention|Hyperinflation|Emphysema",
    "degenerative": r"degenerative|Spondylosis|Osteophyt",
    "fracture": r"Fracture",
}


def mesh_agrees(finding: str, polarity: str, mesh: str) -> bool:
    """Does the manual index take the same side as the sentence? Unindexed studies never qualify."""
    if not mesh or "No Indexing" in mesh:
        return False
    indexed = bool(re.search(MESH_KEYS[finding], mesh, re.I))
    return indexed if polarity == "yes" else not indexed


def _shards() -> tuple[HfFileSystem, list[str]]:
    fs = HfFileSystem()
    return fs, sorted(fs.glob(DATASET_GLOB))


def fetch_index(force: bool = False) -> list[dict[str, Any]]:
    """Every report's text columns, read by column over HTTP; images are not touched."""
    if config.OPENI_INDEX_PATH.exists() and not force:
        return json.loads(config.OPENI_INDEX_PATH.read_text(encoding="utf-8"))
    fs, shards = _shards()
    rows: list[dict[str, Any]] = []
    for shard, path in enumerate(shards):
        with fs.open(path, "rb") as handle:
            parquet = pq.ParquetFile(handle)
            for group in range(parquet.metadata.num_row_groups):
                table = parquet.read_row_group(group, columns=list(TEXT_COLUMNS))
                for offset, row in enumerate(table.to_pylist()):
                    row.update(shard=shard, row_group=group, row=offset)
                    rows.append(row)
                print(f"  shard {shard} group {group}: {len(rows)} reports", flush=True)
    config.OPENI_INDEX_PATH.write_text(json.dumps(rows), encoding="utf-8")
    return rows


def positions(report: dict[str, Any]) -> dict[str, claims.Mention]:
    """What one report asserts or denies, by finding."""
    return claims.resolve(claims.scan(f"{report.get('findings') or ''} {report.get('impression') or ''}"))


def candidates(index: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Per finding, the studies with an explicit yes and the studies with an explicit no."""
    buckets: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"yes": [], "no": []})
    for report in index:
        for finding, mention in positions(report).items():
            if (mention.polarity in ("yes", "no") and not claims.is_comparative(mention.sentence)
                    and mesh_agrees(finding, mention.polarity, report.get("MeSH") or "")):
                buckets[finding][mention.polarity].append(
                    {"uid": report["uid"], "sentence": mention.sentence, "source": mention.source,
                     "mesh": report.get("MeSH") or "", "shard": report["shard"],
                     "row_group": report["row_group"], "row": report["row"],
                     "indication": report.get("indication") or ""})
    return buckets


def author(index: list[dict[str, Any]], seed: int = RNG_SEED) -> tuple[list[dict[str, Any]], dict[str, tuple[int, int]]]:
    """Pick balanced candidates per finding, scarcest finding first, one study each."""
    buckets = candidates(index)
    counts = {f: (len(b["yes"]), len(b["no"])) for f, b in buckets.items()}
    rng = random.Random(seed)
    used: set[int] = set()
    picked: list[dict[str, Any]] = []
    for finding in sorted(buckets, key=lambda f: min(counts[f])):
        if min(counts[finding]) < config.MIN_PER_SIDE:
            continue
        sides = {answer: [c for c in buckets[finding][answer] if c["uid"] not in used] for answer in ("yes", "no")}
        # Spares: some studies have no frontal image, and the hand check deletes.
        n = min(len(sides["yes"]), len(sides["no"]), config.MAX_PER_SIDE + 4)
        for answer in ("yes", "no"):
            rng.shuffle(sides[answer])
            for candidate in sides[answer][:n]:
                used.add(candidate["uid"])
                picked.append({**candidate, "finding": finding, "answer": answer,
                               "phrase": FINDINGS[finding].phrase, "case_id": f"CXR-{candidate['uid']:04d}"})
    return picked, counts


def _fetch_frontals(wanted: list[dict[str, Any]]) -> dict[int, bytes | None]:
    """Frontal images for the chosen studies, one row-group read per group touched."""
    fs, shards = _shards()
    by_group: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for item in wanted:
        by_group[(item["shard"], item["row_group"])].append(item)
    out: dict[int, bytes | None] = {}
    for (shard, group), items in sorted(by_group.items()):
        with fs.open(shards[shard], "rb") as handle:
            table = pq.ParquetFile(handle).read_row_group(group, columns=["uid", "img_frontal"])
        frontals = table.column("img_frontal").to_pylist()
        for item in items:
            out[item["uid"]] = frontals[item["row"]]
        print(f"  fetched {len(items)} frontals from shard {shard} group {group}", flush=True)
    return out


def materialise(picked: list[dict[str, Any]], index: list[dict[str, Any]],
                rebalance: bool = True) -> list[dict[str, Any]]:
    """Write case folders for studies with a frontal image; rebalance; assign probe ids."""
    frontals = _fetch_frontals(picked)
    reports = {r["uid"]: r for r in index}
    kept: list[dict[str, Any]] = []
    for item in picked:
        data = frontals.get(item["uid"])
        if not data:
            continue
        png = images.prepare_bytes(data)
        folder = config.CASES_DIR / item["case_id"]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "frontal.png").write_bytes(png)
        report = reports[item["uid"]]
        casebuild.save(item["case_id"], files={"frontal": (folder / "frontal.png", "cxr", "current")},
                       context=report.get("indication") or "",
                       reference={k: report.get(k) or "" for k in ("findings", "impression", "MeSH", "Problems")})
        kept.append(item)
    return _balance(kept) if rebalance else kept


def _balance(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Trim each finding to equal yes/no counts, capped, then number the probes."""
    per: dict[str, dict[str, list]] = defaultdict(lambda: {"yes": [], "no": []})
    for item in items:
        per[item["finding"]][item["answer"]].append(item)
    out: list[dict[str, Any]] = []
    for finding in sorted(per):
        n = min(len(per[finding]["yes"]), len(per[finding]["no"]), config.MAX_PER_SIDE)
        if n < config.MIN_PER_SIDE:
            continue
        out += per[finding]["yes"][:n] + per[finding]["no"][:n]
    for number, item in enumerate(out, start=1):
        item["pid"] = f"p{number:03d}"
        item.setdefault("checked_at", None)
    return out


def shadow(scored: list[dict[str, Any]], index: list[dict[str, Any]], seed: int = RNG_SEED) -> list[dict[str, Any]]:
    """Unmentioned-as-negative probes on already-fetched studies. Reported, never scored."""
    reports = {r["uid"]: r for r in index}
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for finding in sorted({p["finding"] for p in scored}):
        silent = [p for p in scored if finding not in positions(reports[p["uid"]])]
        rng.shuffle(silent)
        for item in silent[:SHADOW_PER_FINDING]:
            out.append({"pid": f"s{len(out) + 1:03d}", "uid": item["uid"], "case_id": item["case_id"],
                        "finding": finding, "answer": "no", "phrase": FINDINGS[finding].phrase,
                        "source": "unmentioned", "sentence": "", "mesh": item["mesh"]})
    return out


def write(scored: list[dict[str, Any]], shadows: list[dict[str, Any]], counts: dict[str, tuple[int, int]]) -> None:
    """The committed probe file."""
    payload = {"seed": RNG_SEED, "counts": {f: list(c) for f, c in counts.items()},
               "probes": scored, "shadow": shadows}
    config.PROBES_PATH.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def load() -> dict[str, Any]:
    """The probe file, or a pointed error."""
    if not config.PROBES_PATH.exists():
        raise SystemExit("no probes yet - run `python -m src.main build` first")
    return json.loads(config.PROBES_PATH.read_text(encoding="utf-8"))


def missing_images(payload: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Probes whose frontal image is not on disk."""
    payload = payload or load()
    seen: dict[int, dict[str, Any]] = {}
    for probe in payload["probes"] + payload.get("shadow", []):
        if not (config.CASES_DIR / probe["case_id"] / "frontal.png").exists():
            seen.setdefault(probe["uid"], probe)
    return list(seen.values())


def rematerialise(index: list[dict[str, Any]] | None = None) -> int:
    """Re-fetch the images the committed probe file names, without re-authoring it.

    The probe definitions are ours and are committed; the images are IU-Xray's
    and are not. A fresh clone has probes.json and no pixels, so this rebuilds
    the case folders for exactly the studies already chosen - same uids, same
    labels, same digest.
    """
    wanted = missing_images()
    if not wanted:
        return 0
    index = index if index is not None else fetch_index()
    materialise(wanted, index, rebalance=False)
    return len(wanted)


def approve(drop: list[str]) -> dict[str, Any]:
    """The hand check's only two moves: delete a probe, then stamp the rest. Never flip a label."""
    payload = load()
    kept = [p for p in payload["probes"] if p["pid"] not in set(drop)]
    for item in kept:
        item.pop("pid", None)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    balanced = _balance(kept)
    for item in balanced:
        item["checked_at"] = stamp
    payload["probes"] = balanced
    config.PROBES_PATH.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return payload


def prune() -> tuple[int, float]:
    """Delete CXR case folders no probe refers to. Returns (folders, MB reclaimed).

    Authoring over-fetches - spare candidates cover studies with no frontal
    image and probes the hand check deletes - and a re-authored set strands the
    previous one. Four rebuild rounds left 451 orphan folders here, 178 MB, all
    of it re-fetchable. Only CXR-* folders are touched: LAB-*, PAIR-demo and
    anything a reader dropped in by hand are left alone.
    """
    import shutil

    payload = load()
    live = {p["case_id"] for p in payload["probes"]} | {s["case_id"] for s in payload.get("shadow", [])}
    removed, freed = 0, 0
    for folder in config.CASES_DIR.iterdir():
        if not folder.is_dir() or not folder.name.startswith("CXR-") or folder.name in live:
            continue
        freed += sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
        shutil.rmtree(folder)
        removed += 1
    return removed, freed / 2**20


def digest() -> str:
    """sha256 over the probes and the image hashes they point at."""
    payload = load()
    parts = [json.dumps(payload["probes"], sort_keys=True)]
    for probe in payload["probes"]:
        parts.append(images.sha256((config.CASES_DIR / probe["case_id"] / "frontal.png").read_bytes()))
    return "sha256:" + hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


if __name__ == "__main__":
    index = fetch_index()
    picked, counts = author(index)
    print(f"{len(index)} reports; explicit yes/no per finding:")
    for finding, (yes, no) in sorted(counts.items(), key=lambda kv: -min(kv[1])):
        mark = "" if min(yes, no) >= config.MIN_PER_SIDE else "   (dropped: too few on one side)"
        print(f"  {finding:18s} yes {yes:4d}   no {no:4d}{mark}")
    print(f"{len(picked)} candidates picked before image fetch")
