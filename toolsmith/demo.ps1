# The whole thing, in order. Steps 1 and 2 cost nothing.
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

function Step($text) { Write-Host "`n=== $text ===" -ForegroundColor Cyan }
function Smith { $env:PYTHONPATH = "."; uv run --no-sync python -m src.main @args }

Step "1. the guard is code, not a prompt - 19 hostile tool bodies, both layers"
Smith redteam

Step "2. what it starts with: two fixed tools, and four tasks neither can do"
Smith clean
Smith tools
Smith tasks

Step "3. it writes a tool, the guard reads it, the sandbox tests it"
Smith solve gstin_1 --show-source

Step "4. read what it wrote. It is a real file on disk."
Smith inspect

Step "5. the next task in the family: it calls what it already has"
Smith solve gstin_2

Step "6. a different rule, so it writes a second tool and the toolbox grows"
Smith solve isoweek_1
Smith tools
