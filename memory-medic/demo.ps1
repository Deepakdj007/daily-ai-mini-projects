# The whole story in one script. Every date is injected, so the run is
# reproducible and a year of decay happens in about a minute.
$env:PYTHONPATH = "."
function Step($n, $text) { Write-Host "`n=== $n. $text ===" -ForegroundColor Cyan }

Step 0 "a year of remembered facts, seeded as of 1 June"
uv run --no-sync python -m src.main --as-of 2026-06-01 seed --reset

Step 1 "the user announces a move (arrival-time drift)"
uv run --no-sync python -m src.main --as-of 2026-06-10 say "I moved to Bengaluru last week."

Step 2 "a second preference that only holds in another context (the Mem0 trap)"
uv run --no-sync python -m src.main --as-of 2026-06-11 say "For incident calls I'd rather get a phone call." --quiet

Step 3 "two months pass and nobody says anything (silent decay)"
uv run --no-sync python -m src.main --as-of 2026-08-01 sweep --once

Step 4 "a source file changes, unannounced"
(Get-Content data\sources\profile.md) -replace "Zeta Retail", "Nimbus Logistics" |
    Set-Content data\sources\profile.md -Encoding utf8
uv run --no-sync python -m src.main --as-of 2026-08-02 sweep --once

Step 5 "a change to a stable fact: the agent refuses to decide alone"
uv run --no-sync python -m src.main --as-of 2026-08-03 say "My name is actually spelled Rohun Nair." --quiet
uv run --no-sync python -m src.main --as-of 2026-08-03 review

Step 6 "a human settles it"
uv run --no-sync python -m src.main --as-of 2026-08-03 review --decision approve

Step 7 "what was true, and when"
uv run --no-sync python -m src.main --as-of 2026-08-03 timeline demo city
uv run --no-sync python -m src.main --as-of 2026-08-03 timeline demo employer

Step 8 "asking it now"
uv run --no-sync python -m src.main --as-of 2026-08-03 say "Where do I live and who do I work for?"
uv run --no-sync python -m src.main --as-of 2026-08-03 status
