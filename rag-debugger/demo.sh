#!/usr/bin/env bash
# The whole demo in one script:  bash demo.sh
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=.

step() { printf '\n=== %s ===\n' "$1"; }

step "1. free checks first: diagnosis rules and the tracer, no API calls"
uv run python -m src.diagnose
uv run python -m src.tracer

step "2. the retriever on its own: raw similarities for the four questions"
uv run python -m src.retriever

step "3. the debugger: four questions, four different verdicts"
uv run python -m src.main demo

step "4. try the fix the miss suggests, and watch precision drop"
uv run python -m src.main debug "Why does my saree look a different colour than the photo on the website?" --top-k 6
