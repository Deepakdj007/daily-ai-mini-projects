#!/usr/bin/env bash
# The whole argument, in order. Steps 0-3 cost nothing.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=.

step() { printf '\n=== %s ===\n' "$1"; }
smith() { uv run --no-sync python -m src.main "$@"; }

step "0. every check that costs no tokens, before spending any"
uv run --no-sync python -m tools.selftest

step "1. the corpus: six families, and the held-out inputs nobody sees"
smith tasks

step "2. the guard is code, not a prompt - 34 hostile tool bodies, both layers"
smith redteam

step "3. the ladder, and what each rung was predicted to do"
smith ladder

step "4. what it starts with: three fixed tools"
smith clean
smith tools

step "5. the fixed toolset meets a checksum it cannot compute"
smith solve gstin_1 --arm fixed || true

step "6. the same task, allowed to write a tool - guard, tests, hidden inputs"
smith solve gstin_1 --arm persist --show-source || true

step "7. read what it wrote. It is a real file on disk."
smith inspect gstin_check || true

step "8. the next task in the family, toolbox ON: it calls what it already wrote"
smith solve gstin_2 --arm persist || true

step "9. the same task, toolbox OFF: it writes the same tool again and pays twice"
smith solve gstin_2 --arm holdout || true

step "10. a run, and the pre-registered verdict"
smith run --profile smoke
smith report
smith budget
