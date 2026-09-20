"""The only module that talks to Ollama.

Inputs:  a model tag, a prompt, zero or more PNG byte strings, decoding options
Outputs: a Completion with the answer text and what was actually sent

Two rules live here. First, every call sets num_ctx, temperature, top_k and a
seed - Ollama's defaults are silent and wrong for this work. Second, structured
output is a two-stage affair: the model reasons freely with the image, then a
text-only call turns that prose into JSON. Constraining the first call with a
schema suppresses the thinking trace, and Google's own prompts ask for it.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Sequence

import ollama

from src import config

_client: ollama.Client | None = None
_calls = {"count": 0, "images": 0, "seconds": 0.0}


def client() -> ollama.Client:
    """One client for the process."""
    global _client
    if _client is None:
        _client = ollama.Client(host=config.OLLAMA_HOST)
    return _client


def digest(model: str) -> str:
    """The pulled model's digest. Tags move; a results file records the digest."""
    for entry in client().list().models:
        if entry.model == model or (entry.model or "").split(":")[0] == model:
            return entry.digest or ""
    raise SystemExit(f"{model} is not pulled.\n  Fix: ollama pull {model}")


def unload(model: str) -> None:
    """Evict a model from VRAM. Two 4B models will not co-reside on 8 GB."""
    client().generate(model=model, prompt="", keep_alive=0)


def strip_thinking(text: str) -> str:
    """Drop the <unused94>...<unused95> trace MedGemma may emit unprompted.

    Measured on Ollama: the model usually emits a bare `<unused94>` and then
    the answer, with no closing token at all - `'<unused94> ready.'`. Treating
    the open token as "a trace started" and cutting there returns an empty
    string for every call. So: a closed trace is removed; a lone open token is
    deleted and the rest kept. A trace truncated mid-thought shows up as
    `truncated=True` on the Completion instead.
    """
    if config.THINK_CLOSE in text:
        text = text.split(config.THINK_CLOSE, 1)[1]
    # The model also echoes its own image delimiters into prose when it refers
    # to "the first image" - cosmetic, but they would land in the rendered note.
    return _SPECIAL.sub("", text).strip()


_SPECIAL = re.compile(r"<(?:unused\d+|start_of_image|end_of_image)>")


def final_answer(text: str) -> str | None:
    """Whatever follows the last 'Final Answer:', without the reasoning above it."""
    parts = re.split(re.escape(config.FINAL_ANSWER), text, flags=re.IGNORECASE)
    return parts[-1].strip() if len(parts) > 1 else None


@dataclass(frozen=True, slots=True)
class Completion:
    """One model answer plus the facts a gate needs to trust it."""

    text: str
    raw: str
    model: str
    seed: int
    latency_s: float
    prompt_tokens: int
    eval_tokens: int
    image_bytes_sent: int
    truncated: bool


def complete(model: str, prompt: str, images: Sequence[bytes] = (), *,
             seed: int = config.SEED_MAIN, num_ctx: int = config.NUM_CTX_SINGLE,
             num_predict: int = config.NUM_PREDICT_PROBE,
             schema: dict[str, Any] | None = None, system: str = "",
             allow_positional: bool = False) -> Completion:
    """One chat turn. Greedy, seeded, with the context size stated.

    Two or more images require one `[img]` placeholder each in the prompt.
    Ollama swaps them for `[img-0]`, `[img-1]` in order; naming the images in
    prose is what stops the model conflating them (ollama#13586). Only the
    preflight, which measures what happens without labels, may opt out.
    """
    if len(images) > 1 and not allow_positional and prompt.count("[img]") != len(images):
        raise ValueError(f"{len(images)} images need {len(images)} [img] placeholders")
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    user: dict[str, Any] = {"role": "user", "content": prompt}
    if images:
        user["images"] = list(images)
    messages.append(user)

    started = time.perf_counter()
    response = client().chat(
        model=model, messages=messages, format=schema, keep_alive=config.KEEP_ALIVE,
        options={"temperature": config.TEMPERATURE, "top_k": config.TOP_K, "seed": seed,
                 "num_ctx": num_ctx, "num_predict": num_predict})
    latency = time.perf_counter() - started
    raw = response.message.content or ""
    _calls["count"] += 1
    _calls["images"] += len(images)
    _calls["seconds"] += latency
    return Completion(
        text=strip_thinking(raw), raw=raw, model=model, seed=seed, latency_s=latency,
        prompt_tokens=response.prompt_eval_count or 0, eval_tokens=response.eval_count or 0,
        image_bytes_sent=sum(len(image) for image in images),
        truncated=(response.done_reason == "length"))


def structure(model: str, prose: str, schema: dict[str, Any], *,
              instruction: str, num_predict: int = config.NUM_PREDICT_EXTRACT) -> Any:
    """Stage two: turn free text into schema-valid JSON, with no image in the call."""
    prompt = f"{instruction}\n\nTEXT:\n{prose}"
    out = complete(model, prompt, schema=schema, num_predict=num_predict)
    return parse_json(out.text)


def parse_json(text: str) -> Any:
    """Tolerant parse: the whole thing, else the first balanced object or list, else None."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("[", "]"), ("{", "}")):
        start, end = cleaned.find(opener), cleaned.rfind(closer)
        if 0 <= start < end:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                continue
    return None


def stats() -> dict[str, float]:
    """Call counters for the results manifest."""
    return dict(_calls)


if __name__ == "__main__":
    assert strip_thinking("<unused94>hmm<unused95> Final Answer: yes") == "Final Answer: yes"
    assert strip_thinking("<unused94> ready.") == "ready.", "a lone open token must not eat the answer"
    assert strip_thinking("Prior: <start_of_image> <end_of_image> clear") == "Prior:   clear"
    assert final_answer("reasoning...\nFinal Answer: no") == "no"
    assert final_answer("no anchor") is None
    assert parse_json('```json\n[{"a": 1}]\n```') == [{"a": 1}]
    assert parse_json('Final Answer: [{"box_2d": [1,2,3,4], "label": "x"}]')[0]["label"] == "x"
    assert parse_json("nothing here") is None
    print(f"subject digest {digest(config.SUBJECT_MODEL)[:19]}...")
    out = complete(config.SUBJECT_MODEL, "Reply with the single word: ready.", num_predict=8)
    print(f"{out.model} said {out.text!r} in {out.latency_s:.1f}s "
          f"({out.prompt_tokens} prompt tokens, truncated={out.truncated})")
    print("OK - llm")
