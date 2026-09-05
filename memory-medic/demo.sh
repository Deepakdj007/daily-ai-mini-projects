#!/usr/bin/env bash
# The whole story in one script. Every date is injected, so the run is
# reproducible and a year of decay happens in about a minute.
#
#   bash demo.sh
set -euo pipefail
cd "$(dirname "$0")"

step() { printf '\n\033[36m=== %s ===\033[0m\n' "$1"; }
mm() { uv run --no-sync python -m src.main "$@"; }

step "0. a year of remembered facts, seeded as of 1 June"
mm --as-of 2026-06-01 seed --reset

step "1. the user announces a move (arrival-time drift)"
mm --as-of 2026-06-10 say "I moved to Bengaluru last week."

step "2. a second preference that only holds in another context (the Mem0 trap)"
mm --as-of 2026-06-11 say "For incident calls I'd rather get a phone call." --quiet

step "3. two months pass and nobody says anything (silent decay)"
mm --as-of 2026-08-01 sweep --once

step "4. a source file changes, unannounced"
sed -i 's/Zeta Retail/Nimbus Logistics/' data/sources/profile.md
mm --as-of 2026-08-02 sweep --once

step "5. a change to a stable fact: the agent refuses to decide alone"
mm --as-of 2026-08-03 say "My name is actually spelled Rohun Nair." --quiet
mm --as-of 2026-08-03 review

step "6. a human settles it"
mm --as-of 2026-08-03 review --decision approve

step "7. what was true, and when"
mm --as-of 2026-08-03 timeline demo city
mm --as-of 2026-08-03 timeline demo employer

step "8. asking it now"
mm --as-of 2026-08-03 say "Where do I live and who do I work for?"
mm --as-of 2026-08-03 status

printf '\nNow open the inbox:\n  CLOCK_AT=2026-08-03 uv run --no-sync streamlit run src/inbox.py\n'
