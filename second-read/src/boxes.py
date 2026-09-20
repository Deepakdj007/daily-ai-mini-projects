"""Localization: ask where a structure is, parse the box, draw it, test its stability.

Inputs:  a prepared PNG, an object name, a model
Outputs: Boxes in normalised coordinates, and overlay PNGs

There is no gold box in this project - Chest ImaGenome is credentialed - so no
IoU here is against truth. `stability()` compares the model with itself across
seeds, and `swap_sensitivity()` asks whether the box moves when the patient
does. A box that stays put on a stranger's film is a prior over where hearts
usually are, not a reading of this film.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src import config, images, llm, prompts
from src.types import Box

# What to ask the model to put a box around. A finding name is a diagnosis; the
# localization prompt wants a thing with edges.
LOCATE_NAMES = {
    "cardiomegaly": "cardiac silhouette", "pleural_effusion": "pleural effusion",
    "consolidation": "area of consolidation", "pulmonary_edema": "pulmonary edema",
    "pneumothorax": "pneumothorax", "nodule": "pulmonary nodule", "fracture": "rib fracture",
    "atelectasis": "area of atelectasis",
}


def _coerce(values: list[Any]) -> tuple[float, float, float, float] | None:
    """Notebook says 0-1000, paper says 0-1. If anything exceeds 1, it is the former."""
    try:
        y0, x0, y1, x1 = (float(v) for v in values[:4])
    except (TypeError, ValueError):
        return None
    if max(y0, x0, y1, x1) > 1.0:
        y0, x0, y1, x1 = (v / 1000.0 for v in (y0, x0, y1, x1))
    y0, y1 = sorted((min(max(y0, 0.0), 1.0), min(max(y1, 0.0), 1.0)))
    x0, x1 = sorted((min(max(x0, 0.0), 1.0), min(max(x1, 0.0), 1.0)))
    return y0, x0, y1, x1


def parse(text: str) -> list[Box]:
    """Boxes from the Final Answer, or from the first JSON list in the text."""
    payload = llm.parse_json(llm.final_answer(text) or text)
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return []
    out: list[Box] = []
    for item in payload:
        if isinstance(item, dict) and isinstance(item.get("box_2d"), list):
            coords = _coerce(item["box_2d"])
            if coords:
                out.append(Box(*coords, label=str(item.get("label", ""))))
    return out


def locate(model: str, png: bytes, object_name: str, seed: int = config.SEED_MAIN) -> tuple[list[Box], llm.Completion]:
    """Google's prompt, greedy, with the image. Returns the boxes and the completion."""
    out = llm.complete(model, prompts.localize(object_name), (png,), seed=seed,
                       num_predict=config.NUM_PREDICT_BOX)
    return parse(out.text), out


def stability(model: str, png: bytes, object_name: str) -> tuple[list[Box], list[Box], float | None]:
    """Same image, two seeds. Greedy decoding should give an IoU near 1."""
    first, _ = locate(model, png, object_name, seed=config.SEED_MAIN)
    second, _ = locate(model, png, object_name, seed=config.SEED_REPEAT)
    iou = first[0].iou(second[0]) if first and second else None
    return first, second, iou


def swap_sensitivity(model: str, png: bytes, donor_png: bytes, object_name: str) -> tuple[list[Box], list[Box], float | None]:
    """Same prompt on this film and on a stranger's. A high IoU means the box did not look."""
    own, _ = locate(model, png, object_name)
    other, _ = locate(model, donor_png, object_name)
    iou = own[0].iou(other[0]) if own and other else None
    return own, other, iou


def overlay(png: bytes, boxes: list[Box], case_id: str, object_name: str) -> Path:
    """Write output/overlays/<case>-<object>.png."""
    slug = "".join(ch if ch.isalnum() else "-" for ch in object_name.lower()).strip("-")
    return images.draw_boxes(png, boxes, config.OVERLAYS_DIR / f"{case_id}-{slug}.png")


if __name__ == "__main__":
    text = ('The heart sits centrally. Final Answer: ```json[{"box_2d": [350, 300, 700, 750], '
            '"label": "heart"}]```')
    got = parse(text)
    assert len(got) == 1 and abs(got[0].y0 - 0.35) < 1e-9 and got[0].label == "heart", got
    assert parse('[{"box_2d": [0.1, 0.2, 0.3, 0.4], "label": "x"}]')[0].x1 == 0.4, "0-1 scale kept"
    assert parse('[{"box_2d": [700, 750, 350, 300], "label": "x"}]')[0].y0 == 0.35, "swapped corners fixed"
    assert parse("no boxes here") == []
    print("OK - boxes")
