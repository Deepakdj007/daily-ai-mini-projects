# Deep Research Team

A multi-agent research team built on LlamaIndex Workflows. Give it a topic and a
planner splits it into sub-questions, four researchers search the web in parallel,
a reflector checks whether anything important is missing and can send the team
back out for a second round, and a writer produces a cited markdown report.

Runs free. Gemini `gemini-3.1-flash-lite` on the Google AI Studio free tier, and
DuckDuckGo search that needs no API key at all — so the only thing you sign up
for is one Gemini key.

## Setup

```bash
uv sync
cp .env.example .env   # paste a free key from https://aistudio.google.com/apikey
```

## Run

bash:

```bash
PYTHONPATH=. uv run python -m src.main "the state of solid-state batteries in 2026"
```

PowerShell:

```powershell
$env:PYTHONPATH="."; uv run python -m src.main "the state of solid-state batteries in 2026"
```

With no topic it falls back to a default one. The report lands in `output/report.md`.

Check the search tool on its own, before you spend a single token:

```bash
PYTHONPATH=. uv run python -m src.search
```

## See the parallelism

The researchers run concurrently by default. Drop them to one and compare:

```bash
PYTHONPATH=. RESEARCH_WORKERS=1 uv run python -m src.main "how lithium-ion battery recycling works at industrial scale"
```

On the same topic: **11.4s** with 4 researchers, **24.9s** with 1. With four you
also see the `done` lines arrive out of order, which is the fan-out actually
working rather than just being configured.

## How it flows

```
StartEvent -> plan -> (fan out N) -> research -> (fan in) -> gather
                                        ^                      |
                                        |                      v
                                        +----- reflect <-------+
                                                  |
                                                  v
                                                write -> StopEvent
```

Fan-out is `ctx.send_event` in a loop. Fan-in is `ctx.collect_events`, which
returns `None` on every call until the last expected event lands. `reflect` is
what closes the loop, and `MAX_ROUNDS` in `src/config.py` is what stops it.

## Project layout

```
src/
├── config.py     keys, model ids, caps, make_llm()
├── events.py     every Event + the pydantic models
├── prompts.py    the four prompts
├── search.py     keyless DuckDuckGo, off the event loop
├── llm.py        structured + plain-text model calls
├── workflow.py   the five steps
└── main.py       CLI, live event stream, report writer
```

## Tuning

Everything worth changing is in `src/config.py`: `NUM_SUB_QUESTIONS`,
`MAX_ROUNDS`, `SEARCH_RESULTS`, `RESEARCH_WORKERS`, and `WORKFLOW_TIMEOUT`.
Note that the workflow default timeout is 45 seconds, which a real run exceeds —
this project sets 600.
