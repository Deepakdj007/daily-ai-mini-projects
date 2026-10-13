# The whole demo in one script.
#
#   powershell -ExecutionPolicy Bypass -File demo.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONPATH = "."

function Step($t) { Write-Host "`n=== $t ===" -ForegroundColor Cyan }

# PowerShell 5.1 ignores a native command's exit code even with "Stop", so check it by hand.
# Otherwise a failed free check would still go on to spend Groq tokens.
function Run { uv run python @args; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } }

Step "1. free checks first: diagnosis rules and the tracer, no API calls"
Run -m src.diagnose
Run -m src.tracer

Step "2. the retriever on its own: raw similarities for the four questions"
Run -m src.retriever

Step "3. the debugger: four questions, four different verdicts"
Run -m src.main demo

Step "4. try the fix the miss suggests, and watch precision drop"
Run -m src.main debug "Why does my saree look a different colour than the photo on the website?" --top-k 6
