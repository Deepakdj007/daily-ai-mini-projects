"""Multi-image asks: the prior-versus-current comparison and the CT slice stack.

Inputs:  two chest films, or a NIfTI volume, and a model
Outputs: the model's prose, a structured summary, and the completion

Both put several images in one message, so both label every image in the
prompt with an [img] slot and text between them. Neither is scored against
ground truth in this project: MS-CXR-T and every labelled CT set are
credentialed. The comparison ships with a prior-swap control instead; the CT
read ships labelled as demonstrated, not measured.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src import config, images, llm, prompts

CHANGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"changes": {"type": "array", "items": {
        "type": "object",
        "properties": {"finding": {"type": "string"},
                       "change": {"type": "string",
                                  "enum": ["improved", "worsened", "stable", "new", "resolved", "not_assessable"]}},
        "required": ["finding", "change"]}}},
    "required": ["changes"],
}


def compare(model: str, prior_png: bytes, current_png: bytes) -> tuple[str, list[dict], llm.Completion]:
    """Prior then current, each named in the prompt. Returns prose, per-finding changes, completion."""
    out = llm.complete(model, prompts.LONGITUDINAL, (prior_png, current_png),
                       num_predict=config.NUM_PREDICT_NOTE)
    parsed = llm.structure(model, out.text, CHANGE_SCHEMA, instruction=prompts.LONGITUDINAL_STRUCTURE,
                           num_predict=256)
    changes = parsed.get("changes", []) if isinstance(parsed, dict) else []
    return out.text, [c for c in changes if isinstance(c, dict)], out


def prior_swap(model: str, current_png: bytes, real_prior: bytes, stranger_prior: bytes) -> dict[str, Any]:
    """The same comparison twice: with the real prior and with a stranger's.

    Any interval change the model reports against a stranger's film is a
    comparison it made up. The count of such survivors is the honest residue
    of a capability nobody can benchmark for free.
    """
    _, real, _ = compare(model, real_prior, current_png)
    _, swapped, _ = compare(model, stranger_prior, current_png)
    by_finding = lambda rows: {r["finding"]: r["change"] for r in rows}  # noqa: E731
    real_map, swapped_map = by_finding(real), by_finding(swapped)
    survivors = [f for f, change in real_map.items()
                 if change not in ("stable", "not_assessable") and swapped_map.get(f) == change]
    return {"real": real_map, "swapped": swapped_map, "survivors": survivors}


def ct_read(model: str, nii_path: Path, history: str, label: str,
            n_slices: int = config.CT_SLICES_DEFAULT) -> tuple[str, str | None, llm.Completion]:
    """The paper's CT prompt over an equidistant slice stack. Returns prose, final answer, completion."""
    slices = images.ct_slices(nii_path, n_slices)
    # Alternate the edge by one pixel so no two adjacent slices share a size (ollama#17321).
    slices = [s if k % 2 == 0 else images.prepare_bytes(s, edge=config.IMAGE_EDGE - 1)
              for k, s in enumerate(slices)]
    out = llm.complete(model, prompts.ct(len(slices), history, label), slices,
                       num_ctx=config.NUM_CTX_STACK, num_predict=512)
    return out.text, llm.final_answer(out.text), out


if __name__ == "__main__":
    demo = config.DEMO_DIR
    before, after = demo / "longitudinal_before.png", demo / "longitudinal_after.png"
    if before.exists() and after.exists():
        prior = images.prepare(before, edge=config.IMAGE_EDGE - 1)
        current = images.prepare(after)
        text, changes, out = compare(config.SUBJECT_MODEL, prior, current)
        print(f"comparison in {out.latency_s:.1f}s ({out.prompt_tokens} prompt tokens):")
        print("  " + " ".join(text.split())[:400])
        print(f"  structured: {changes}")
    print("OK - multi")
