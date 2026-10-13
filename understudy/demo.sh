#!/usr/bin/env bash
# The whole thing on the sample chat. Costs nothing and needs no GPU.
set -u
cd "$(dirname "$0")"

step() { printf '\n\033[36m=== %s ===\033[0m\n' "$1"; }
u() { PYTHONPATH=. uv run --no-sync python -m src.main "$@"; }

step "1. who is in this export, and did the dates parse the way you expect"
u inspect

step "2. the export becomes training pairs - watch what gets thrown away"
u build --me "Arjun Mehta"

step "3. what did this person actually say that could be asked about later"
u facts scan --me "Arjun Mehta"

step "4. the gates: a question that copies the chat proves nothing"
u facts check --me "Arjun Mehta"

step "5. what the fine-tune moved, and what it did not"
u score --me "Arjun Mehta"

step "6. read the misses yourself - the grader cannot see a paraphrase"
u misses --me "Arjun Mehta"
