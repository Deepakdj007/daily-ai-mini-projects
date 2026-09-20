# second-read, the whole argument in eleven steps.
# Run from the project root: .\demo.ps1
$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "."

function Step($n, $what) { Write-Host "`n=== $n. $what ===" -ForegroundColor Cyan }

Step 1 "The models are pulled, and what a call costs"
uv run python -m src.main models

Step 2 "Three things nobody has published: image merging, the canary, VRAM"
uv run python -m src.preflight merge canary vram

Step 3 "Build: 20 synthetic lab reports, 100 authored probes, every case folder"
uv run python -m src.main build

Step 4 "Every validity check that costs no inference"
uv run python -m src.main gates

Step 5 "The four audit conditions, and what was predicted before the run"
uv run python -m src.main ladder

Step 6 "The product with the audit switched OFF - what you would have shipped"
uv run python -m src.main note CXR-0050

Step 7 "The same case, audited: every claim re-asked four ways"
uv run python -m src.main case CXR-0050

Step 8 "A lab report page to structured JSON, scored exactly, plus the blind floor"
uv run python -m src.main case LAB-01

Step 9 "Localization: the box drawn on the film, its stability, and whether it moves patient to patient"
uv run python -m src.main locate CXR-0050 --finding cardiomegaly

Step 10 "A CT volume as a slice stack - demonstrated, not measured"
uv run python -m src.main ct --slices 12

Step 11 "The leaderboard: gates first, then the pre-registered verdict"
uv run python -m src.main report

Write-Host "`nnote      output\notes\CXR-0050.md" -ForegroundColor Green
Write-Host "overlay   output\overlays\CXR-0050-cardiomegaly.png" -ForegroundColor Green
Write-Host "results   output\leaderboard.md" -ForegroundColor Green
