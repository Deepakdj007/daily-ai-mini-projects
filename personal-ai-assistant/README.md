# Personal AI Assistant

A tool-calling agent that lives in your Telegram chat and does real things: web search, personal notes, reminders that fire on their own, and summaries of any article, YouTube video, or PDF you send it. It also messages you first every morning with a briefing — nobody has to ask for it.

Built on Groq `openai/gpt-oss-120b` (free tier) with `langchain`/`langgraph`'s `create_agent`. Every inbound message goes to the LLM first, which decides intent and calls the right tool — that router is the whole lesson. Conversation memory and reminders both survive a restart: LangGraph's `AsyncSqliteSaver` and a plain SQLite table are the source of truth, not anything held in memory.

Works on WhatsApp too, with the Business API — the agent core (`src/agent.py`, `src/tools.py`) doesn't know or care which chat platform is calling it. Telegram just needs a free `@BotFather` token instead of a Meta Business account and template approval, so that's what this build uses.

## Setup

```bash
uv sync
cp .env.example .env   # paste a BotFather token + a free Groq key
```

Get a bot token from [@BotFather](https://t.me/BotFather) (`/newbot`), and a free Groq key from <https://console.groq.com/keys>.

## Run

```bash
# bash
PYTHONPATH=. uv run python -m src.main
```
```powershell
# PowerShell
$env:PYTHONPATH="."; uv run python -m src.main
```

Then in Telegram:

1. Send `/start` — the bot replies with your chat id.
2. Paste it into `.env` as `OWNER_CHAT_ID`, then restart the bot. This is what stops anyone else who finds the bot from using your free Groq quota.
3. Talk to it normally, or try `/help` and `/briefing`.

## Project layout

```
src/
  config.py       API keys, model, paths, timezone, make_llm()
  store.py        sqlite3: notes + reminders
  readers.py      article (trafilatura) / YouTube transcript / PDF (pypdf) -> plain text
  tools.py        @tool functions the agent routes to
  google_tools.py optional Calendar + Gmail tools (disabled without credentials.json)
  prompts.py      system prompt + morning-briefing prompt
  agent.py        create_agent() wiring + a safe ask() wrapper
  scheduler.py    JobQueue: daily briefing, one-off reminders, restart rehydration
  main.py         Telegram handlers + entrypoint
```

## Optional: Calendar + Gmail

By default the bot runs with zero Google setup. To add Calendar read/create and Gmail read/draft:

```bash
uv add google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

Create an OAuth client ID (Desktop app) in [Google Cloud Console](https://console.cloud.google.com/apis/credentials), download it as `credentials.json` into this folder, and restart the bot — it will open a browser to authorize once and cache the result in `token.json`. Email tools only ever create a draft; the bot never sends mail on your behalf.

## Tuning

- `BRIEFING_HOUR` / `BRIEFING_TOPICS` / `TIMEZONE` in `src/config.py` — when the morning message fires and what it searches for.
- `MAX_TOOL_CHARS` — how much of an article/transcript/PDF reaches the model. Groq's free tier is capped at roughly 200k tokens/day, org-wide, so this matters more than the 131k context window.
