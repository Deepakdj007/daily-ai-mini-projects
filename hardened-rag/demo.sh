#!/usr/bin/env bash
# The whole argument in one script, on a single question.
#
#   bash demo.sh
set -euo pipefail
cd "$(dirname "$0")"

step() { printf '\n\033[36m=== %s ===\033[0m\n' "$1"; }
hr() { PYTHONPATH=. uv run --no-sync python -m src.main "$@"; }

step "0. build the index: the clean corpus plus every attack, in one store"
hr build

step "1. the checks that cost nothing, before anything is spent"
hr gates

step "2. the ladder, and what each rung was predicted to do"
hr ladder

step "3. clean retrieval: the pipeline works, and nothing is wrong yet"
hr ask q01 --arm naive --cond clean

step "4. three poisoned copies arrive. The tutorial pipeline believes them"
hr ask q01 --arm naive --cond poison-p

step "5. a cross-encoder does not help - it promotes the poison"
hr ask q01 --arm rerank --cond poison-p

step "6. the injection classifier catches the passage that gives orders"
hr ask q01 --arm guard --cond inject-overt

step "7. ... and misses the one that just states a false figure"
hr ask q01 --arm guard --cond inject-policy

step "8. reading each passage alone: no longer fooled, no longer useful"
hr ask q01 --arm isolate --cond poison-p

step "9. the textbook fix, voting, loses to three copies"
hr ask q01 --arm majority --cond poison-p

step "10. deciding from metadata the model never saw"
hr ask q01 --arm provenance --cond poison-p

step "11. and where that stops working: same tier, newer date"
hr ask q01 --arm provenance --cond sametier

printf '\nNow run the ladder:\n  PYTHONPATH=. uv run python -m src.main run --profile lite\n'
printf '  PYTHONPATH=. uv run python -m src.main report\n'
