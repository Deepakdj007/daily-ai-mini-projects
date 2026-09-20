#!/usr/bin/env bash
# second-read, the whole argument in eleven steps.
set -euo pipefail
export PYTHONPATH=.

step() { printf "\n\033[36m=== %s ===\033[0m\n" "$1"; }

step "1. The models are pulled, and what a call costs"
uv run python -m src.main models

step "2. Three things nobody has published: image merging, the canary, VRAM"
uv run python -m src.preflight merge canary vram

step "3. Build: 20 synthetic lab reports, 100 authored probes, every case folder"
uv run python -m src.main build

step "4. Every validity check that costs no inference"
uv run python -m src.main gates

step "5. The four audit conditions, and what was predicted before the run"
uv run python -m src.main ladder

step "6. The product with the audit switched OFF - what you would have shipped"
uv run python -m src.main note CXR-0050

step "7. The same case, audited: every claim re-asked four ways"
uv run python -m src.main case CXR-0050

step "8. A lab report page to structured JSON, scored exactly, plus the blind floor"
uv run python -m src.main case LAB-01

step "9. Localization: the box, its stability, and whether it moves patient to patient"
uv run python -m src.main locate CXR-0050 --finding cardiomegaly

step "10. A CT volume as a slice stack - demonstrated, not measured"
uv run python -m src.main ct --slices 12

step "11. The leaderboard: gates first, then the pre-registered verdict"
uv run python -m src.main report

printf "\n\033[32mnote      output/notes/CXR-0050.md\noverlay   output/overlays/CXR-0050-cardiomegaly.png\nresults   output/leaderboard.md\033[0m\n"
