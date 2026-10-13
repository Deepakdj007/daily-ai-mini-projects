# The whole thing on the sample chat. Costs nothing and needs no GPU.
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

function Step($text) { Write-Host "`n=== $text ===" -ForegroundColor Cyan }
function U { $env:PYTHONPATH = "."; uv run --no-sync python -m src.main @args }

Step "1. who is in this export, and did the dates parse the way you expect"
U inspect

Step "2. the export becomes training pairs - watch what gets thrown away"
U build --me "Arjun Mehta"

Step "3. what did this person actually say that could be asked about later"
U facts scan --me "Arjun Mehta"

Step "4. the gates: a question that copies the chat proves nothing"
U facts check --me "Arjun Mehta"

Step "5. what the fine-tune moved, and what it did not"
U score --me "Arjun Mehta"

Step "6. read the misses yourself - the grader cannot see a paraphrase"
U misses --me "Arjun Mehta"
