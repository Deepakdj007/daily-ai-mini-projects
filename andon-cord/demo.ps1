# The whole argument in one script.
#
#   powershell -ExecutionPolicy Bypass -File demo.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Step($t) { Write-Host "`n=== $t ===" -ForegroundColor Cyan }
function Cord { $env:PYTHONPATH = "."; uv run --no-sync python -m src.main @args }

$ambiguous = "Put together a competitive brief on our autopay flow for the leadership review next week."
$complete  = "Compare UPI AutoPay and eNACH for recurring payments in India: mandate limits, validity periods, and which one a subscription business should pick."
$answer    = "The merchant dashboard, codenamed Kestrel, and the audience is the Hyderabad risk committee."

Step "0. every check that costs no tokens, before spending any"
$env:PYTHONPATH = "."; uv run --no-sync python -m tools.selftest

Step "1. search works on its own, with no API key"
$env:PYTHONPATH = "."; uv run --no-sync python -m src.search

Step "2. a complete brief: the team never stops to ask"
"" | Cord run $complete --ask team

Step "3. the same team, a brief that hides something: the lead pulls the cord"
Cord run $ambiguous --ask team --park

Step "4. the question survives the process that asked it"
Cord inbox

Step "5. answer it from here, and the whole line moves again"
Write-Host "Paste this when prompted:" -ForegroundColor Yellow
Write-Host "  $answer"
Write-Host "Then run: PYTHONPATH=. uv run python -m src.main answer <thread_id>"

Step "6. cut the cord and the same brief is researched on a guess"
"" | Cord run $ambiguous --ask never

Step "7. and the honest number: how often does it catch the ambiguity?"
Cord ask-rate --samples 5 --all

Write-Host "`nNow the end-to-end comparison, both policies over four briefs:"
Write-Host "  PYTHONPATH=. uv run python -m src.main compare"
