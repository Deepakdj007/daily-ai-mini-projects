# Approval Gate Agent

A LangGraph agent that drafts an irreversible action — an email, a calendar event, or a support ticket — then **freezes itself before executing** and routes the draft to a human for approve / edit / reject. It uses `interrupt()` to pause mid-graph and a durable `SqliteSaver` checkpointer, so the pending approval survives a process restart: you can close the terminal, come back later, and resume the exact same draft. A deterministic validator pre-checks every draft, a revision limit stops the re-draft loop from spinning forever, and every step is written to an append-only audit log.

## Setup

```bash
uv init approval-gate-agent
uv add langgraph langgraph-checkpoint-sqlite langchain-groq pydantic python-dotenv
cp .env.example .env   # then paste a free key from https://console.groq.com/keys
```

## Run

Draft an action and freeze it for approval (this process exits; the state is on disk):

```bash
PYTHONPATH=. uv run python -m src.main propose "Email priya@acme.com that the Q3 launch slips two weeks"
```

Resume it later — even from a brand-new terminal — using the printed `thread_id`:

```bash
PYTHONPATH=. uv run python -m src.main review <thread_id>
# then type: approve   |   edit <note>   |   reject <reason>
```

List threads still awaiting approval:

```bash
PYTHONPATH=. uv run python -m src.main list
```

Executed actions are recorded in `output/ledger.json`; the full decision trail is in `output/audit_log.jsonl`.
