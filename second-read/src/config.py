"""Central configuration: paths, model tags, decoding settings, and every threshold.

Inputs:  environment variables from .env (OLLAMA_HOST only - there is no API key)
Outputs: module-level constants imported by every other module

Every number that moves a reported figure lives here with the reasoning next to
it, so the leaderboard can point at one file when it says what it ran with.
Nothing in this project needs a key or a credit card: the models run on the
laptop through Ollama, and the only network calls are one-time data pulls.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# On Windows `uv run` does not inherit the shell's environment, so this call is
# mandatory and must come before any os.getenv below.
load_dotenv()


def _force_utf8_console() -> None:
    """Stop Windows cp1252 consoles crashing on model output.

    Radiology prose from a model routinely carries en-dashes, degree signs and
    curly quotes. On a default Windows console those raise UnicodeEncodeError
    mid-run and kill a long pass for nothing.
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


_force_utf8_console()

# --- Paths -------------------------------------------------------------------
# Derived from this file, never from the working directory, so every command
# behaves the same no matter where it is launched from.
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CASES_DIR = DATA_DIR / "cases"
REFERENCE_DIR = DATA_DIR / "reference"
DEMO_DIR = DATA_DIR / "demo"
OUTPUT_DIR = ROOT_DIR / "output"
CASE_RECORDS_DIR = OUTPUT_DIR / "cases"
NOTES_DIR = OUTPUT_DIR / "notes"
OVERLAYS_DIR = OUTPUT_DIR / "overlays"
LEADERBOARD_PATH = OUTPUT_DIR / "leaderboard.md"
PROBES_PATH = REFERENCE_DIR / "probes.json"
OPENI_INDEX_PATH = REFERENCE_DIR / "openi_index.json"
LAB_TRUTH_PATH = REFERENCE_DIR / "lab_truth.json"

for _dir in (CASES_DIR, REFERENCE_DIR, DEMO_DIR, CASE_RECORDS_DIR, NOTES_DIR, OVERLAYS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


def results_path(model: str) -> Path:
    """Per-model results file, so the twin's run never overwrites the subject's."""
    return OUTPUT_DIR / f"results-{model.replace('/', '-').replace(':', '-')}.json"


# --- Models ------------------------------------------------------------------
# Optional override for a non-default Ollama server (None -> localhost:11434).
OLLAMA_HOST = os.getenv("OLLAMA_HOST") or None

# All three are Q4_K_M on ollama.com, 3.3 GB each, and all accept images.
# gemma3:4b is the same decoder without the medical pretraining, which makes it
# the control that isolates what the medical tuning bought. medgemma:4b is
# MedGemma 1 and is accepted by --model but is not part of the default run.
SUBJECT_MODEL = os.getenv("SUBJECT_MODEL", "medgemma1.5:4b")
TWIN_MODEL = os.getenv("TWIN_MODEL", "gemma3:4b")
LEGACY_MODEL = "medgemma:4b"
# 5.0 GB. Pulled once for the quantisation experiment: nobody has published
# whether Q4 degrades MedGemma's box coordinates, so the preflight measures it.
Q8_MODEL = "medgemma1.5:4b-it-q8_0"

# --- Images ------------------------------------------------------------------
# The model card normalises every image to 896x896 and encodes it to 256 tokens.
# Ollama does not implement pan-and-scan, so a 2,500 px film and an 896 px film
# reach the model as the same 256 tokens. Resizing first makes the bytes sent
# equal the bytes hashed, so a cache key describes what the model actually saw.
IMAGE_EDGE = 896
IMAGE_TOKENS = 256

# --- Decoding ----------------------------------------------------------------
# Ollama allocates 4,096 context by default on a GPU under 24 GB and the
# medgemma1.5 Modelfile sets only a stop token, so the advertised 128K is a
# capability, not an allocation. Two images plus a prompt plus reasoning does
# not fit in 4,096, and the overflow is silent. Every call sets num_ctx.
NUM_CTX_SINGLE = 8192
NUM_CTX_STACK = 16384
TEMPERATURE = 0.0
TOP_K = 1
SEED_MAIN = 0
SEED_REPEAT = 1
NUM_PREDICT_PROBE = 256
NUM_PREDICT_NOTE = 1024
NUM_PREDICT_BOX = 1000       # what Google's localization notebook uses
NUM_PREDICT_EXTRACT = 2048   # a 12-row lab table as a JSON list
KEEP_ALIVE = "10m"

# MedGemma may think before answering even when not asked to. The trace sits
# between these two tokens and the answer follows the second one. The model
# card does not mention them; the official notebooks do.
THINK_OPEN = "<unused94>"
THINK_CLOSE = "<unused95>"
FINAL_ANSWER = "Final Answer:"

# --- The audit ---------------------------------------------------------------
CONDITIONS = ("sighted", "blind", "shuffled", "repeat")

# Above this share of repeat-disagreements the sighted answers are not
# reproducible, and a flag computed from a coin flip is not a finding. The run
# is then VOID rather than failed.
INSTABILITY_CEILING = 0.10

# --- The criterion's numbers -------------------------------------------------
# At temperature zero a blind call is a constant per finding, so "the model
# said this blind too" is true of every claim on the default side of a finding
# and a flag-rate bar would be met by construction. The claim that can fail is
# that flagged claims are WRONG more often than cleared ones: fifteen points is
# the gap below which a reader gains nothing from the tag, and at ~110 probes
# split roughly in half Fisher's exact test resolves a gap of that size at 0.05.
CRITERION_GAP = 0.15
CRITERION_P = 0.05
# Probes are balanced per finding with identical wording on both sides, so a
# blind model that guesses any fixed answer per finding scores exactly 50%.
# Anything outside this band is a leak, not a result.
BLIND_FLOOR = 0.50
BLIND_TOLERANCE = 0.05
CANARY_SIGHTED_FLOOR = 0.90
CANARY_FRAMES = 16

# --- Probe authoring ---------------------------------------------------------
# Fewer than six explicit assertions or six explicit negations and a finding's
# balanced set is too small to say anything; it is dropped, not padded.
MIN_PER_SIDE = 6
# Warm probe calls measured at ~2 s, so eight a side over seven findings is
# 112 probes and under 20 minutes per model at four conditions each.
MAX_PER_SIDE = 8
# Thirty probes, sighted against blind: under an 8-point gap there is no
# evidence signal to ablate and the project reports that and stops.
PILOT_N = 30
PILOT_MIN_GAP = 0.08

# --- Lab reports -------------------------------------------------------------
N_LAB_REPORTS = 20
# Rendered narrow at modest DPI. At 300 DPI a full page is 2,550 px wide and
# the 896 px squash destroys the digits; the table has to survive that squash.
LAB_DPI = 150

# --- CT ----------------------------------------------------------------------
# The paper's preprocessing: every axial slice rescaled to 896x896 with three
# HU windows stacked as colour channels, up to 85 slices sampled equidistantly.
# Twelve slices is 3,072 vision tokens, which an 8 GB card handles.
CT_SLICES_DEFAULT = 12
CT_WINDOWS_HU = ((-1024, 1024), (-135, 215), (0, 80))

# --- Cache-key versions ------------------------------------------------------
PROMPT_VERSION = "v1"

if __name__ == "__main__":
    print(f"root            {ROOT_DIR}")
    print(f"subject         {SUBJECT_MODEL}")
    print(f"twin            {TWIN_MODEL}")
    print(f"image edge      {IMAGE_EDGE}px -> {IMAGE_TOKENS} tokens")
    print(f"num_ctx         {NUM_CTX_SINGLE} single / {NUM_CTX_STACK} stack")
    print(f"decoding        temperature {TEMPERATURE}, top_k {TOP_K}, seeds {SEED_MAIN}/{SEED_REPEAT}")
    print(f"criterion       flagged claims >= {CRITERION_GAP:.0%} less accurate, Fisher p < {CRITERION_P}")
    print(f"blind band      {BLIND_FLOOR - BLIND_TOLERANCE:.0%} .. {BLIND_FLOOR + BLIND_TOLERANCE:.0%}")
    print(f"ollama host     {OLLAMA_HOST or 'default (127.0.0.1:11434)'}")
