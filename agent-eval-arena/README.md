# Agent Eval Arena

A head-to-head evaluation harness for tool-calling agents: the same calculator-tool agent runs against a fixed 10-item test set on three Groq models (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `openai/gpt-oss-20b`), every call is traced in Langfuse, each answer is scored by exact-match and an LLM judge, and the results are aggregated into a leaderboard with accuracy, cost, and latency per model.

## Setup

```bash
uv init agent-eval-arena
uv add langfuse openai rich python-dotenv
cp .env.example .env   # paste a free Groq key + free Langfuse Cloud keys
```

## Run

```bash
PYTHONPATH=. uv run python src/main.py
```

Prints a Rich leaderboard table, writes `leaderboard.md`, and every item run is traced as a Langfuse experiment — check the printed dataset run URLs to inspect individual traces.
