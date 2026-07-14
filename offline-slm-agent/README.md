# offline-slm-agent

A tool-calling AI agent that runs 100% offline on your laptop. A small language model
(`qwen3:4b`, ~2.6 GB) served by Ollama reasons across multiple steps and calls six local
tools — calculator, notes (write/read/list), clock, word counter. No API key, no network,
no credit card.

```
you -> agent loop -> qwen3:4b (local) -> tool calls -> results fed back -> final answer
```

`qwen3:4b` is the smallest model that chains tool calls dependably. Lighter options like
`llama3.2:3b` (~2 GB) run faster but fumble multi-step tool schemas — they'll calculate a
result and then just claim they saved it without actually calling the save tool. `qwen3:1.7b`
is worth trying if RAM is tight, with the same caveat.

## Setup

1. Install Ollama: https://ollama.com/download (make sure you're on a recent version —
   `winget upgrade Ollama.Ollama` if `ollama pull` reports a version error)
2. Pull the model (one-time, ~2.6 GB):

```bash
ollama pull qwen3:4b
```

3. Install Python deps:

```bash
uv sync
```

## Run

```bash
# bash / Git Bash
PYTHONPATH=. uv run python -m src.main

# PowerShell
$env:PYTHONPATH="."; uv run python -m src.main
```

## Example prompts

- `Calculate 15% of 1240, then save the result to a note called tip.txt`
- `What notes do I have, and what's in tip.txt?`
- `What time is it, and how many words are in 'hello world from my laptop'?`

Turn on airplane mode and try them again — everything still works.
