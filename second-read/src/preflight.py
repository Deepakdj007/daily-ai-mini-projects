"""Three measurements nobody has published, run before any product code trusts them.

Inputs:  a pulled model and the demo images
Outputs: printed results, to be copied into FINDINGS.md as they happen

1. Does Ollama merge two same-size images into one? (ollama#17321 is confirmed
   for Qwen-VL and unconfirmed for the Gemma 3 architecture. It decides how the
   longitudinal pair and the CT stack have to be sent.)
2. Does Q4 quantisation move MedGemma's bounding boxes? (No data exists.)
3. What does VRAM actually look like at each context size and image count?
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Sequence

from src import config, images, llm, prompts


def _ask(model: str, prompt: str, frames: Sequence[bytes], num_ctx: int = config.NUM_CTX_SINGLE) -> str:
    out = llm.complete(model, prompt, frames, num_ctx=num_ctx, num_predict=200, allow_positional=True)
    return " ".join(out.text.split())[:300]


def image_merge(model: str = config.SUBJECT_MODEL) -> None:
    """Send two frames three ways and read back how many the model saw."""
    a, b = images.burn_text("FRAME ALPHA"), images.burn_text("FRAME BRAVO")
    b_short = images.burn_text("FRAME BRAVO", edge=config.IMAGE_EDGE - 1)
    question = ("How many separate images were you given, and what text is printed "
                "in each one? End with 'Final Answer: <number>'.")
    trials = {
        "positional, same size": (question, (a, b)),
        "[img] labelled, same size": (f"First image: [img]\nSecond image: [img]\n{question}", (a, b)),
        "[img] labelled, sizes differ by 1px": (f"First image: [img]\nSecond image: [img]\n{question}", (a, b_short)),
    }
    print(f"\n[1] two-image handling on {model}")
    for label, (prompt, frames) in trials.items():
        answer = _ask(model, prompt, frames)
        seen = llm.final_answer(answer) or "?"
        print(f"  {label:38s} -> final answer {seen!r}\n      {answer}")


def canary(model: str = config.SUBJECT_MODEL) -> None:
    """Sighted must read the burned-in text; blind must not. Proves the image arrives.

    What matters is whether the true id appears, not whether the model answered.
    Blind, it will happily invent one - the gate scores the string, not the fact
    that a Final Answer line exists.
    """
    ident = "KQ-7741"
    frame = images.burn_text(f"CASE ID {ident}")
    sighted = _ask(model, prompts.CANARY, (frame,))
    blind = _ask(model, prompts.CANARY, ())
    print(f"\n[canary] the id printed on the frame is {ident}")
    for label, answer in (("sighted", sighted), ("blind", blind)):
        read = ident.lower() in answer.lower()
        print(f"  {label:8s} said {llm.final_answer(answer)!r}  -> reads the id: {read}"
              + ("   <- must be True" if label == "sighted" else "   <- must be False"))


def vram(model: str = config.SUBJECT_MODEL) -> None:
    """Measured residency at each context size and image count, via `ollama ps`."""
    print(f"\n[3] VRAM residency for {model}")
    frames = [images.burn_text(f"SLICE {k}") for k in range(8)]
    for num_ctx in (config.NUM_CTX_SINGLE, config.NUM_CTX_STACK):
        for count in (1, 8):
            shown = frames[:count]
            prompt = "".join(f"Slice {k + 1}: [img]\n" for k in range(count)) if count > 1 else ""
            prompt += "How many slices were you given? Reply 'Final Answer: <number>'."
            llm.unload(model)
            out = llm.complete(model, prompt, shown, num_ctx=num_ctx, num_predict=40)
            status = llm.client().ps()
            for proc in status.models:
                if (proc.model or "").startswith(model.split(":")[0]):
                    total = proc.size or 0
                    on_gpu = proc.size_vram or 0
                    split = f"{on_gpu / 2**30:.2f} GB GPU / {(total - on_gpu) / 2**30:.2f} GB CPU"
                    print(f"  num_ctx {num_ctx:>5}  images {count}  prompt {out.prompt_tokens:>5} tok  "
                          f"{out.latency_s:5.1f}s  {split}  said {llm.final_answer(out.text)!r}")


def boxes_q4_vs_q8(finding: str = "cardiac silhouette", n_films: int = 6) -> None:
    """The same localization prompt at two quantisations, on several films, scored against each other.

    Model-major: all films at q4, unload, all films at q8. Two 4B models do not
    share this card.
    """
    import json

    from src import boxes

    films = {"wikimedia": images.prepare(config.DEMO_DIR / "cxr_wikimedia.png")}
    if config.PROBES_PATH.exists():
        pool = json.loads(config.PROBES_PATH.read_text(encoding="utf-8"))["probes"]
        for probe in [p for p in pool if p["finding"] == "cardiomegaly" and p["answer"] == "yes"][: n_films - 1]:
            films[probe["case_id"]] = (config.CASES_DIR / probe["case_id"] / "frontal.png").read_bytes()
    print(f"\n[2] localization of {finding!r} on {len(films)} films: q4 vs q8")
    results: dict[str, dict[str, list]] = {}
    for model in (config.SUBJECT_MODEL, config.Q8_MODEL):
        llm.unload(config.SUBJECT_MODEL)
        llm.unload(config.Q8_MODEL)
        results[model] = {}
        for name, png in films.items():
            found, out = boxes.locate(model, png, finding)
            results[model][name] = found
            print(f"  {model:26s} {name:10s} {out.latency_s:5.1f}s  "
                  f"{[tuple(round(v, 2) for v in (b.y0, b.x0, b.y1, b.x1)) for b in found]}")
    ious = []
    for name in films:
        q4, q8 = results[config.SUBJECT_MODEL][name], results[config.Q8_MODEL][name]
        iou = q4[0].iou(q8[0]) if q4 and q8 else None
        ious.append(iou)
        print(f"  {name:10s} IoU q4 vs q8: {iou if iou is None else f'{iou:.3f}'}")
    valid = [i for i in ious if i is not None]
    if valid:
        valid.sort()
        print(f"  median IoU {valid[len(valid) // 2]:.3f} over {len(valid)} films, min {valid[0]:.3f}, max {valid[-1]:.3f}")


if __name__ == "__main__":
    import sys

    which = sys.argv[1:] or ["merge", "canary", "vram"]
    if "merge" in which:
        image_merge()
    if "canary" in which:
        canary()
    if "vram" in which:
        vram()
    if "boxes" in which:
        boxes_q4_vs_q8()
