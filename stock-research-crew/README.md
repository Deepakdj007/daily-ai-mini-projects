# Stock Research Crew

A parallel multi-agent equity research team built on CrewAI. Give it a ticker and
four analysts study the stock at the same time — one reads the chart, one reads the
filings, one reads the news, one reads the sell side — then a Head of Equity Research
fuses their four briefings into a written note and saves it to disk.

The interesting part is the join. Each analyst carries exactly one tool, so it only
ever sees its own slice of data and its context stays small. The editor carries no
tools at all: it works purely from what the four handed back. And because the analysts
have no dependency on each other, they run concurrently — which is a single flag, not
a graph.

Runs free. Gemini `gemini-3.1-flash-lite` on the Google AI Studio free tier, and
yfinance for market data, which needs no API key at all — so the only thing you sign
up for is one Gemini key.

## Setup

```bash
uv sync
cp .env.example .env   # paste a free key from https://aistudio.google.com/apikey
```

## Run

bash:

```bash
PYTHONPATH=. uv run python -m src.main RELIANCE.NS
```

PowerShell:

```powershell
$env:PYTHONPATH="."; uv run python -m src.main RELIANCE.NS
```

With no ticker it defaults to `RELIANCE.NS`. The note lands in
`output/<TICKER>-report.md`.

Indian listings need their exchange suffix — `RELIANCE.NS` and `TCS.NS` for NSE,
`.BO` for BSE. Bare `RELIANCE` returns nothing, so the CLI checks the symbol resolves
and exits before spending a single token:

```
Cannot analyse RELIANCE: no price data for 'RELIANCE'.
Indian listings need a suffix — try RELIANCE.NS or RELIANCE.BO.
```

US tickers work as-is (`AAPL`, `NVDA`), though `returnOnEquity` and analyst coverage
are richer for large caps than for small ones.

Check the data layer on its own, before you spend a single token:

```bash
PYTHONPATH=. uv run python -m src.market RELIANCE.NS
```

That prints all four blocks exactly as the agents will see them, with a character and
token count under each — useful when you want to know what a run will cost.

## See the parallelism

The four analysts run concurrently by default. Force them into single file and
compare:

```bash
PYTHONPATH=. uv run python -m src.main RELIANCE.NS --sequential
```

Same ticker, same five agents, same models, measured on `RELIANCE.NS`:

| Mode | Wall clock |
|---|---|
| 4 analysts in parallel | **8.9s**, 11.6s, 14.3s |
| 4 analysts one at a time | **20.5s**, 24.4s |

Roughly 2x, and the gap is almost entirely the three analysts that no longer wait
their turn. Individual runs vary because free-tier latency does — so the honest claim
is the range, not a single best number. With `verbose=True` you also see the four tool
calls fire before any of them return, which is the fan-out actually working rather
than just being configured.

One run in that sample took 34s, because Gemini's free tier returned
`503 - This model is currently experiencing high demand` to one analyst and CrewAI
retried the step. The note came out complete. If you see a run take noticeably longer
than the rest, that is usually what happened — `MAX_RETRIES` in `src/config.py` is
what absorbs it.

## How it flows

```
                    ┌─ Technical Analyst    (async) ─┐
                    ├─ Fundamental Analyst  (async) ─┤
kickoff(ticker) ────┤                                ├──→ Head of Equity Research
                    ├─ News Analyst         (async) ─┤    (sync, context=[all 4])
                    └─ Consensus Analyst    (async) ─┘              │
                                                                    ▼
                                                    output/<TICKER>-report.md
```

Fan-out is `async_execution=True` on the four analyst tasks. Fan-in is `context=[...]`
on the editor task, which is what makes it wait for all four.

The editor task must stay synchronous. CrewAI validates that a crew ends with at most
one asynchronous task, so making the editor async fails the whole run with a
`ValidationError` before any work starts.

## Project layout

```
src/
├── config.py     keys, model ids, rate limits, make_llm()
├── fmt.py        rupee/crore formatting, safe statement row access
├── yahoo.py      talking to Yahoo — cached Ticker, validate(), profile()
├── market.py     the four compact text blocks the agents actually read
├── tools.py      the four @tool wrappers, one per analyst
├── agents.py     the five agents
├── tasks.py      the five tasks — four async, one sync join
├── crew.py       assembles the crew, times the run
└── main.py       CLI, ticker validation, writes the note
```

## Why the data layer does so much work

`Ticker.info` alone is 166 keys, and the three financial statements come back as full
DataFrames. Handing that to four agents would blow both the free tier and the context
window, so every function in `src/market.py` reduces its data to a short labelled
block — roughly 100 to 600 tokens each instead of tens of thousands.

It also never raises. A missing field is completely normal (Yahoo leaves
`returnOnEquity` empty for plenty of Indian listings, so the code derives it from net
income and equity instead), and a dead ticker should produce a readable
`DATA UNAVAILABLE` line rather than a traceback three agents deep.

## Tuning

Everything worth changing is in `src/config.py`:

| Setting | Default | What it does |
|---|---|---|
| `MODEL_ANALYST` | `gemini-3.1-flash-lite` | Raise to `gemini-3.5-flash` if briefings come back thin |
| `MODEL_EDITOR` | `gemini-3.1-flash-lite` | The model that writes the note |
| `MAX_RPM` | `12` | Per-agent request ceiling; four analysts fire at once |
| `MAX_RETRIES` | `3` | Retries per agent — the free tier returns 503 often enough to need them |
| `HISTORY_PERIOD` | `1y` | How much price history the technical analyst sees |
| `NEWS_COUNT` | `8` | Headlines fetched — the largest single token cost |
| `MAX_ITER` | `5` | Each analyst needs one tool call; more means it is thrashing |

## Not investment advice

Every note carries a disclaimer, and it is meant literally. This is a demonstration of
multi-agent orchestration that happens to use market data. The models summarise figures
they are given, they do not verify them, and nothing here has been reviewed by a
registered adviser.
