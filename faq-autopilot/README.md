# FAQ Autopilot

A long-horizon agent that watches a folder of source docs, detects when one changes, and edits a customer FAQ on its own — every answer grounded in a source document, every action written to an audit log with the reason it fired. No prompting each time: it runs continuously and reacts to events, the way a teammate would.

Built on Groq `openai/gpt-oss-120b` (free tier) with strict `json_schema` structured output. State lives in SQLite so the agent survives restarts. `FAQ.md` and `CHANGELOG.md` are rendered from the database on every pass.

## Setup

```bash
uv sync
cp .env.example .env   # paste a free key from https://console.groq.com/keys
```

## Run

Single pass (bootstraps the FAQ from `docs/`, then exits):

```bash
# bash
PYTHONPATH=. uv run python -m src.main --reset --once
# PowerShell
$env:PYTHONPATH="."; uv run python -m src.main --reset --once
```

Always-on watcher (edit any file in `docs/` and watch it react; Ctrl-C to stop):

```bash
PYTHONPATH=. uv run python -m src.main --watch --interval 3
```

Then open `FAQ.md` (the maintained answers) and `CHANGELOG.md` (what the agent did and why).
