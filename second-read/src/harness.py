"""Running the probe set: every probe, every condition, every model - model-major, resumably.

Inputs:  a profile name and the committed probe set
Outputs: output/results-<model>.json - one row per probe, plus canary, lab and shadow rows

Model-major because two 4B models will not share 8 GB: all of one model's
work, an explicit unload, then the next. Probes run first so a stopped run can
still answer the question the project is about. Rows already on disk are
skipped, so a run interrupted at row 40 resumes at row 41.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from src import audit, casebuild, config, gates, images, llm, probes, prompts, workstation
from src.types import Polarity

# `lab_models` is separate from `models` on purpose. The synthetic lab pages
# are the INSTRUMENT that proves the blind condition really removes the
# evidence - a floor of zero that a chest film cannot provide - and that only
# needs establishing for the model under test. Nothing in the report reads the
# twin's lab rows, and running them costs about two hours: gemma3 has no
# medical tuning to stop it, so it fills all 2,048 output tokens on every one
# of the four calls a page takes, roughly six times the subject's cost.
PROFILES: dict[str, dict[str, Any]] = {
    "smoke": {"models": (config.SUBJECT_MODEL,), "lab_models": (config.SUBJECT_MODEL,),
              "probes": 6, "lab": 2, "shadow": 0, "canary": 4},
    "lite": {"models": (config.SUBJECT_MODEL,), "lab_models": (config.SUBJECT_MODEL,),
             "probes": None, "lab": None, "shadow": None, "canary": config.CANARY_FRAMES},
    "full": {"models": (config.SUBJECT_MODEL, config.TWIN_MODEL),
             "lab_models": (config.SUBJECT_MODEL,),
             "probes": None, "lab": None, "shadow": None, "canary": config.CANARY_FRAMES},
}


def _probe_row(model: str, probe: dict, pool: list[dict], conditions=config.CONDITIONS) -> dict[str, Any]:
    """Sighted first; the donor is chosen against the sighted answer, then the rest."""
    png = (config.CASES_DIR / probe["case_id"] / "frontal.png").read_bytes()
    sighted = audit.ask(model, probe["phrase"], png, condition="sighted", seed=config.SEED_MAIN,
                        evidence_shown=("frontal",))
    lean: Polarity = sighted.polarity if sighted.polarity in ("yes", "no") else probe["answer"]
    donor = workstation.donor_png(probe["finding"], lean, probe["case_id"], pool)
    rest = audit.run(model, probe["phrase"], png, "frontal", donor, [c for c in conditions if c != "sighted"])
    answers = {"sighted": sighted, **rest}
    result = audit.audit(probe["pid"], probe["finding"], sighted.polarity, answers, reference=probe["answer"])
    return {
        "kind": "probe", "pid": probe["pid"], "case_id": probe["case_id"], "finding": probe["finding"],
        "answer": probe["answer"], "source": probe["source"],
        **{c: answers[c].polarity for c in answers},
        "verdict": result.verdict, "flagged": result.flagged, "correct": result.correct,
        "donor": donor[0] if donor else None,
        "image_bytes": {c: answers[c].image_bytes_sent for c in answers},
        "truncated": any(a.truncated for a in answers.values()),
        "latency_s": round(sum(a.latency_s for a in answers.values()), 2),
        "reasoning": {c: audit.reasoning(answers[c].raw)[:300] for c in answers},
    }


def _canary_rows(model: str, n: int) -> list[dict[str, Any]]:
    """Burned-in ids: sighted must read them, blind must not."""
    rows = []
    for k in range(n):
        ident = f"KQ-{1000 + k * 37}"
        frame = images.burn_text(f"CASE ID {ident}")
        seen = llm.complete(model, prompts.CANARY, (frame,), num_predict=40)
        blind = llm.complete(model, prompts.CANARY, (), num_predict=40)
        rows.append({"kind": "canary", "pid": f"k{k:02d}", "id": ident,
                     "sighted_read": ident.lower() in seen.text.lower(),
                     "blind_read": ident.lower() in blind.text.lower(),
                     "image_bytes": {"sighted": seen.image_bytes_sent, "blind": blind.image_bytes_sent}})
    return rows


def _lab_row(model: str, case_id: str) -> dict[str, Any]:
    record = workstation.read_lab(model, casebuild.load(case_id))
    return {"kind": "lab", "pid": case_id, "score": record["sighted"]["score"],
            "sighted_recovered": record["sighted"]["values_recovered"],
            "blind_recovered": record["blind"]["values_recovered"],
            "blind_prose_collisions": record["blind"]["prose_collisions"],
            "image_bytes": {"sighted": record["sighted"]["image_bytes_sent"],
                            "blind": record["blind"]["image_bytes_sent"]}}


def _write(model: str, rows: list[dict], profile: str, stopped: str = "") -> dict[str, Any]:
    payload = {"manifest": {
        "model": model, "digest": llm.digest(model), "profile": profile, "probes": probes.digest(),
        "prompt_version": config.PROMPT_VERSION, "temperature": config.TEMPERATURE, "top_k": config.TOP_K,
        "seeds": [config.SEED_MAIN, config.SEED_REPEAT], "num_ctx": config.NUM_CTX_SINGLE,
        "image_edge": config.IMAGE_EDGE, "llm_calls": llm.stats(),
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"), "stopped": stopped},
        "results": rows}
    path = config.results_path(model)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return payload


def _load(model: str) -> list[dict]:
    """Previous rows, if they came from this probe set and these prompts."""
    path = config.results_path(model)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest = payload.get("manifest", {})
    if manifest.get("probes") != probes.digest() or manifest.get("prompt_version") != config.PROMPT_VERSION:
        print(f"  {path.name} predates the current probes or prompts - starting over")
        return []
    return payload["results"]


def run(profile: str = "lite", *, resume: bool = True) -> None:
    """The matrix, model by model."""
    failed = gates.failures(gates.run_all())
    if failed:
        raise SystemExit("gates failed:\n  " + "\n  ".join(failed))
    plan = PROFILES[profile]
    payload = probes.load()
    scored = payload["probes"][: plan["probes"]]
    shadows = payload["shadow"][: plan["shadow"]] if plan["shadow"] != 0 else []
    all_labs = casebuild.list_cases("LAB-")[: plan["lab"]]
    pool = payload["probes"]

    for model in plan["models"]:
        labs = all_labs if model in plan["lab_models"] else []
        rows = _load(model) if resume else []
        done = {(r["kind"], r["pid"]) for r in rows}
        todo = ([("probe", p) for p in scored] + [("canary", None)] + [("lab", c) for c in labs]
                + [("shadow", s) for s in shadows])
        todo = [t for t in todo if t[0] == "canary" or (t[0], t[1]["pid"] if t[0] != "lab" else t[1]) not in done]
        print(f"{model}: {len(todo)} items to run, {len(rows)} already recorded", flush=True)
        for number, (kind, item) in enumerate(todo, start=1):
            # `did` is what this step actually produced, which is not always
            # rows[-1]: the canary step is a no-op when canary rows already
            # exist, and printing the tail row there claims work that did not
            # happen.
            did: dict[str, Any] | None = None
            if kind == "probe":
                did = _probe_row(model, item, pool)
                rows.append(did)
            elif kind == "shadow":
                did = {**_probe_row(model, item, pool, conditions=("sighted", "blind")), "kind": "shadow"}
                rows.append(did)
            elif kind == "lab":
                did = _lab_row(model, item)
                rows.append(did)
            elif kind == "canary" and not any(r["kind"] == "canary" for r in rows):
                fresh = _canary_rows(model, plan["canary"])
                rows.extend(fresh)
                did = fresh[-1] if fresh else None
            if number % 5 == 0 or number == len(todo):
                _write(model, rows, profile)
                note = "nothing to do" if did is None else (
                    f"{did['kind']} {did['pid']}  {did.get('verdict') or ('scored' if did.get('score') else '')}")
                print(f"  {number}/{len(todo)}  {note}", flush=True)
        _write(model, rows, profile)
        llm.unload(model)
        print(f"  {model} done and unloaded")


if __name__ == "__main__":
    run("smoke")
