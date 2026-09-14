#!/usr/bin/env bash
# The whole argument in one script.
#
#   bash demo.sh

set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=.

step() { printf '\n=== %s ===\n' "$1"; }
cord() { uv run --no-sync python -m src.main "$@"; }

ambiguous="Put together a competitive brief on our autopay flow for the leadership review next week."
complete="Compare UPI AutoPay and eNACH for recurring payments in India: mandate limits, validity periods, and which one a subscription business should pick."
answer="The merchant dashboard, codenamed Kestrel, and the audience is the Hyderabad risk committee."

step "1. search works on its own, with no API key"
uv run --no-sync python -m src.search

step "2. a complete brief: the team never stops to ask"
cord run "$complete" --ask team </dev/null

step "3. the same team, a brief that hides something: the lead pulls the cord"
cord run "$ambiguous" --ask team --park

step "4. the question survives the process that asked it"
cord inbox

step "5. answer it from here, and the whole line moves again"
echo "Paste this when prompted:"
echo "  $answer"
echo "Then run: PYTHONPATH=. uv run python -m src.main answer <thread_id>"

step "6. cut the cord and the same brief is researched on a guess"
cord run "$ambiguous" --ask never </dev/null

step "7. and the honest number: how often does it catch the ambiguity?"
cord ask-rate --samples 5 --all

echo
echo "Now the end-to-end comparison, both policies over four briefs:"
echo "  PYTHONPATH=. uv run python -m src.main compare"
