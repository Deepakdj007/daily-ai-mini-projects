# The whole argument, in order. Steps 0-3 cost nothing.
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

function Step($text) { Write-Host "`n=== $text ===" -ForegroundColor Cyan }
function Smith { $env:PYTHONPATH = "."; uv run --no-sync python -m src.main @args }

Step "0. every check that costs no tokens, before spending any"
$env:PYTHONPATH = "."; uv run --no-sync python -m tools.selftest

Step "1. the corpus: six families, and the held-out inputs nobody sees"
Smith tasks

Step "2. the guard is code, not a prompt - 34 hostile tool bodies, both layers"
Smith redteam

Step "3. the ladder, and what each rung was predicted to do"
Smith ladder

Step "4. what it starts with: three fixed tools"
Smith clean
Smith tools

Step "5. the fixed toolset meets a checksum it cannot compute"
Smith solve gstin_1 --arm fixed

Step "6. the same task, allowed to write a tool - guard, tests, hidden inputs"
Smith solve gstin_1 --arm persist --show-source

Step "7. read what it wrote. It is a real file on disk."
Smith inspect gstin_check

Step "8. the next task in the family, toolbox ON: it calls what it already wrote"
Smith solve gstin_2 --arm persist

Step "9. the same task, toolbox OFF: it writes the same tool again and pays twice"
Smith solve gstin_2 --arm holdout

Step "10. a run, and the pre-registered verdict"
Smith run --profile smoke
Smith report
Smith budget
