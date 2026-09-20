#!/usr/bin/env bash
# The whole thing, in order. Steps 1 and 2 cost nothing.
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=.

step() { printf '\n=== %s ===\n' "$1"; }
smith() { uv run --no-sync python -m src.main "$@"; }

step "1. the guard is code, not a prompt - 19 hostile tool bodies, both layers"
smith redteam

step "2. what it starts with: two fixed tools, and four tasks neither can do"
smith clean
smith tools
smith tasks

step "3. it writes a tool, the guard reads it, the sandbox tests it"
smith solve gstin_1 --show-source

step "4. read what it wrote. It is a real file on disk."
smith inspect

step "5. the next task in the family: it calls what it already has"
smith solve gstin_2

step "6. a different rule, so it writes a second tool and the toolbox grows"
smith solve isoweek_1
smith tools
