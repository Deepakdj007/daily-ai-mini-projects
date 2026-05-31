# a2a-agents

A minimal multi-agent system built on the A2A (Agent2Agent) protocol. A Research Agent and a Writer Agent each run as A2A-compliant servers that publish an Agent Card, and an orchestrator discovers both and chains them — it sends a topic to Research, then passes that summary to Writer — over JSON-RPC, powered by Groq. See [GUIDE.md](GUIDE.md) for the full walkthrough.

## Run

Set `GROQ_API_KEY` in `.env` (see `.env.example`). Then, in three terminals:

```bash
# Research Agent
PYTHONPATH=. uv run uvicorn agents.research_agent:app --port 8001

# Writer Agent
PYTHONPATH=. uv run uvicorn agents.writer_agent:app --port 8002

# Orchestrator (runs the full chain)
PYTHONPATH=. uv run python orchestrator.py
```
