# The whole argument in one script, on a single question.
#
#   powershell -ExecutionPolicy Bypass -File demo.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Step($t) { Write-Host "`n=== $t ===" -ForegroundColor Cyan }
function Hr { $env:PYTHONPATH = "."; uv run --no-sync python -m src.main @args }

Step "0. build the index: the clean corpus plus every attack, in one store"
Hr build

Step "1. the checks that cost nothing, before anything is spent"
Hr gates

Step "2. the ladder, and what each rung was predicted to do"
Hr ladder

Step "3. clean retrieval: the pipeline works, and nothing is wrong yet"
Hr ask q01 --arm naive --cond clean

Step "4. three poisoned copies arrive. The tutorial pipeline believes them"
Hr ask q01 --arm naive --cond poison-p

Step "5. a cross-encoder does not help - it promotes the poison"
Hr ask q01 --arm rerank --cond poison-p

Step "6. the injection classifier catches the passage that gives orders"
Hr ask q01 --arm guard --cond inject-overt

Step "7. ... and misses the one that just states a false figure"
Hr ask q01 --arm guard --cond inject-policy

Step "8. reading each passage alone: no longer fooled, no longer useful"
Hr ask q01 --arm isolate --cond poison-p

Step "9. the textbook fix, voting, loses to three copies"
Hr ask q01 --arm majority --cond poison-p

Step "10. deciding from metadata the model never saw"
Hr ask q01 --arm provenance --cond poison-p

Step "11. and where that stops working: same tier, newer date"
Hr ask q01 --arm provenance --cond sametier

Write-Host "`nNow run the ladder:"
Write-Host "  PYTHONPATH=. uv run python -m src.main run --profile lite"
Write-Host "  PYTHONPATH=. uv run python -m src.main report"
