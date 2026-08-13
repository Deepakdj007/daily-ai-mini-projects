# night-scout

An ambient agent. You do not prompt it — it wakes on a clock through the night, checks HackerNews, arXiv and two RSS feeds, and decides what is worth your morning. It remembers everything it has already looked at, so it never shows you the same thing twice.

What it does not do is act on its own. When it finds something good it writes the reading-list entry, then stops and parks the draft mid-run. The night process exits. Hours later you open the inbox in your browser, and a completely different process picks those parked runs back up exactly where they froze. You approve, reject, or send one back for a rewrite.

Everything is free and keyless. The three sources need no accounts at all, and the model is Groq's `openai/gpt-oss-120b` on the free tier. A full night costs roughly 25,000 tokens against a 200,000-per-day allowance, and there is a `--stub` mode that runs the whole thing with no API key and no requests.

## Setup

```bash
uv sync
cp .env.example .env      # paste a free key from https://console.groq.com/keys
```

Open `src/config.py` and rewrite `INTEREST_PROFILE` in your own words before the first run. It is the prompt that decides what the agent keeps, and the default describes someone who builds Python AI agents — probably not exactly you.

## Run

Watch a whole night in about three minutes. `--demo` compresses the clock; `--stub` swaps in canned model responses so it costs nothing:

```bash
PYTHONPATH=. uv run python -m src.night run --demo --stub
```
```powershell
$env:PYTHONPATH="."; uv run python -m src.night run --demo --stub
```

Drop `--stub` to use the real model, and drop `--demo` too for an actual overnight run — started before 23:00, it sleeps until the window opens, works until 07:00, and exits.

```bash
PYTHONPATH=. uv run python -m src.night run --demo    # real model, fast clock
PYTHONPATH=. uv run python -m src.night run           # the real thing, overnight
```

Then in the morning — or in another terminal, with the night process finished and gone:

```bash
PYTHONPATH=. uv run streamlit run src/inbox.py
```
```powershell
$env:PYTHONPATH="."; uv run streamlit run src/inbox.py
```

Other commands:

```bash
PYTHONPATH=. uv run python -m src.night status        # what is parked, what happened
PYTHONPATH=. uv run python -m src.night reset         # clear items, drafts, reading list
PYTHONPATH=. uv run python -m src.night reset --all   # also drop the checkpoints
```

### Actually running it while you sleep

Starting it by hand every evening rather defeats the point. On Windows, register it once with Task Scheduler and it starts itself:

```powershell
schtasks /create /tn "night-scout" /sc daily /st 22:55 /tr `
  "cmd /c cd /d D:\path\to\night-scout && set PYTHONPATH=. && uv run python -m src.night run >> night.log 2>&1"
```

22:55 rather than 23:00 because the run waits for the window to open by itself. On macOS or Linux the equivalent is a cron line: `55 22 * * * cd /path/to/night-scout && PYTHONPATH=. uv run python -m src.night run >> night.log 2>&1`.

Keep the project out of OneDrive or any synced folder. Sync clients take their own file handles and can copy a database mid-transaction, which corrupts SQLite's write-ahead log.

## What a night looks like

```
night 20260806 · 11 wakes · demo clock (x240)
window 23:00 to 07:15

  [sim Thu 23:00 | real 08:12:04] polled  50  fresh 12  parked 1  queued 4
  [sim Thu 23:45 | real 08:12:20] polled  50  fresh  2  parked 1  queued 4
  [sim Fri 00:30 | real 08:12:34] polled  50  fresh  0  parked 1  queued 3
  ...
```

`fresh 0` is the memory working — it polled the same 50 items and recognised every one. `parked 1` on a wake that found nothing new is the carry-over queue: winners that could not be drafted earlier, because each wake spends only one draft, get picked up later. `queued` is how many are still waiting their turn.

## How it fits together

```
  night process (exits before you wake up)          your morning
  ----------------------------------------          -----------------------

  clock --> poll HN / arXiv / RSS                   streamlit inbox
              |                                            |
              v                                            |
        drop what it has seen                              |
              |                                            |
              v                                            |
        one batched triage call --> scores                 |
              |                                            |
              v                                            |
        best undrafted winner                              |
              |                                            |
              v                                            |
        detail -> draft -> gate ---- interrupt() ---+      |
                                                    v      v
                                            memory.db -----+--> Approve
                                       (the parked draft)          |
                                                                   v
                                                        output/reading-list.md
```

The gate is a real pause, not a callback. LangGraph writes the half-finished run to `memory.db` and the process is free to exit. `Command(resume=...)` from the inbox continues it.

## Project layout

```
src/
  config.py    every knob: window, cadence, budgets, sources, interest profile
  state.py     Item, the pydantic schemas, and the graph's state
  store.py     sqlite: seen items, the parked-draft index, the wake log
  sources.py   HN, arXiv and RSS, normalized into one Item shape
  triage.py    scores a whole wake's items in a single call
  llm.py       the Groq call: rate throttle and retry ladder
  stub.py      canned responses for --stub, so a night costs nothing
  nodes.py     detail, draft, gate (the interrupt), commit, discard
  graph.py     wiring, the checkpointer, and the thread-id scheme
  night.py     the clock and the CLI
  inbox.py     the Streamlit inbox that resumes parked runs
```

Two databases, on purpose. `memory.db` belongs to LangGraph and holds the parked runs; `nightscout.db` is ours and holds what the agent has seen plus an index of which threads are waiting. The index stores pointers only — a draft's text lives in exactly one place, its checkpoint.

## Tuning

All of it is in `src/config.py`.

| Setting | Default | What it changes |
|---|---|---|
| `INTEREST_PROFILE` | a Python agent builder | What gets kept. Edit this first |
| `SCORE_THRESHOLD` | 7 | How picky it is. 8 is much pickier than it sounds |
| `WINDOW_START_HOUR` / `WINDOW_END_HOUR` | 23 / 7 | When it works |
| `WAKE_EVERY_MINUTES` | 45 | How often it checks |
| `MAX_DEEP_PASSES_PER_WAKE` | 1 | Drafts per wake. Raising it front-loads the night |
| `MAX_DEEP_PASSES_PER_NIGHT` | 6 | The whole night's ceiling |
| `MIN_SECONDS_BETWEEN_LLM_CALLS` | 15 | Raise it if you see 429s |
| `RSS_FEEDS` | 2 feeds | Add your own. Any RSS or Atom URL works |
| `MAX_ITEM_AGE_HOURS` | 48 | Older than this is not news |
| `DEMO_TIME_SCALE` | 240 | How fast `--demo` runs |

## Notes

Sources disagree about everything. HN sends unix timestamps, arXiv sends ISO 8601, RSS sends RFC 822 — so every fetcher converts to unix seconds, because sorting those formats as strings puts `"Wed, 31 May 2023"` ahead of `"2026-08-05"`. The Hugging Face feed serves its whole archive back to 2021 and carries no `<description>` at all. arXiv asks for one request every three seconds and returns a 429 if you ignore it. `sources.py` handles each of these, and a source that fails simply sits out that wake.

This is not a news reader. It is deliberately slow and picky — one draft per wake, six a night, and a human gate in front of the only action it can take.
