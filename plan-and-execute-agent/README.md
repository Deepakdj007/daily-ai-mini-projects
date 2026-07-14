# Plan-and-Execute Agent (vs ReAct)

A LangGraph agent that **plans before it acts**: a *planner* breaks a goal into an
ordered step list, an *executor* works each step with tools (Wikipedia + a
calculator), and a *replanner* revises the remaining steps after every result until
the goal is answered. It runs side-by-side with a plain **ReAct** agent — same model,
same tools — so you can see, in one comparison table, how up-front planning trades
more LLM calls for structure and reliability on complex, multi-hop goals.

Powered by Groq `openai/gpt-oss-120b` (free tier, no credit card). Only a Groq key is
needed — the tools are keyless.

## Setup

```bash
uv sync
cp .env.example .env      # then paste a free key from https://console.groq.com/keys
```

## Run

```bash
# Run both agents on the same goal and print the side-by-side comparison
PYTHONPATH=. uv run python -m src.main "your goal here"

# Or run just one
PYTHONPATH=. uv run python -m src.main --agent plan  "your goal here"
PYTHONPATH=. uv run python -m src.main --agent react "your goal here"
```

With no goal it runs a built-in multi-hop demo (comparing two mountain heights).
