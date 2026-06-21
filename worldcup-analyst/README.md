# World Cup Analyst — Parallel Multi-Agent Briefing (LangGraph + Groq)

A supervisor resolves the focus team's **actual next fixture** (opponent + date),
then fans out to three **tool-using agents** that run **in parallel**, and a
synthesizer fans them back in into a single match **preview**. Each agent is a real
ReAct loop: it **decides which tools to call**, reads the results, and loops until
it can answer — not a node that summarizes a fixed fetch.

Ask `"Give me a briefing on Brazil's next match"` and get back the upcoming
opponent, a **form comparison vs that opponent**, a scouted **key player**, and a
**news & storylines** read (injuries, momentum) — all framed toward the next match
and assembled concurrently.

```
START -> supervisor --Send fan-out--> matchup_agent  ┐ each a ReAct
         (router + resolve            player_agent    │ tool loop,
          team id + fixture)          news_agent     ┘ run in parallel
                                            │
                                            └--> synthesizer --> END
                                                      │
                                       retry once on a TRANSIENT failure ┘
```

The three agents and the tools they choose among:

| Agent | Tools it decides among |
|---|---|
| `matchup_agent` | `get_team_form`, `get_opponent_form`, `get_group_standings` |
| `player_agent` | `get_top_scorers`, `get_player_profile` (TheSportsDB bio) |
| `news_agent` | `get_rss_headlines`, `search_news` (Tavily, agent phrases the query) |

Run with `--verbose` to print each agent's tool calls.

## Data sources (all free, no credit card)

- **football-data.org** — official scores, standings, top scorers (World Cup = `WC`).
- **TheSportsDB** — player bios (position / club / nationality), free demo key.
- **BBC + Guardian football RSS** — keyless news feeds, always on.
- **Tavily** — optional team-targeted news search (1000 free searches/mo).

## Models (free-tier-aware)

Groq rate-limits per model, so we split the work across two buckets:

- **Router + agent tool-loops:** `llama-3.1-8b-instant` — tool selection and
  relaying facts is well within 8b, and it draws from a separate, far larger daily
  token budget, so a briefing's many tool calls stay sustainable.
- **Final synthesis:** `llama-3.3-70b-versatile` for reader-facing prose, with an
  automatic **8b fallback** if the 70b daily budget is spent.

## Stack

- **uv** for everything (no pip/venv).
- **langgraph 1.2.5** — `Send` for parallel fan-out; `Annotated[list, operator.add]`
  reducers so concurrent agent writes merge.
- **langchain-groq 1.1.3** + **langchain-core** tools — `ChatGroq.bind_tools` drives
  a hand-rolled ReAct loop (`app/agents/runner.py`).
- **httpx** async clients (football-data.org, TheSportsDB, RSS); **tavily-python**
  (async) + **feedparser** for news.
- **pydantic 2** schemas for every external object.

## Setup

```bash
uv sync                       # install dependencies
cp .env.example .env          # then fill in the two keys below
```

### Environment variables (`.env`)

| Variable | Required | Where to get it (free, no credit card) |
|---|---|---|
| `GROQ_API_KEY` | yes | https://console.groq.com/keys |
| `FOOTBALL_DATA_TOKEN` | yes for live data | https://www.football-data.org/client/register (free token by email) |
| `TAVILY_API_KEY` | optional | https://app.tavily.com — richer team news; without it the news agent uses RSS only |
| `LANGSMITH_API_KEY` | optional | https://smith.langchain.com — tracing turns on only if set |

Without `FOOTBALL_DATA_TOKEN` the graph still runs end-to-end — each worker
degrades gracefully and the briefing reports which sections were unavailable. The
news agent works with no extra key (BBC + Guardian RSS); Tavily just makes it
sharper and more team-specific.

## Run it

```bash
PYTHONPATH=. uv run python app/main.py "Give me a briefing on Brazil's next match"
```

Omit the query to use the built-in sample query.

## Project layout

```
app/
  config.py         # env + per-task models (router/agents on 8b, synthesis on 70b)
  state.py          # TypedDict state + operator.add reducers for parallel writes
  data/client.py    # async football-data.org client (TTL-cached) + next match
  data/news.py      # async news client: BBC + Guardian RSS + optional Tavily
  data/sportsdb.py  # async TheSportsDB client (player bios)
  data/models.py    # pydantic schemas (match, standing, scorer, player, news)
  data/results.py   # shared ApiResult (+ transient flag) + error explainer
  agents/
    supervisor.py   # router + resolves team id and next fixture
    tools.py        # data layer exposed as LangChain tools (per-agent factories)
    runner.py       # the ReAct tool loop + 429 backoff (makes a worker an agent)
    matchup.py      # form + opponent form + standings agent
    player.py       # key-player agent (scorers -> bio)
    news.py         # news agent (RSS + Tavily)
    synthesizer.py  # fan-in: writes the final briefing
  graph.py          # build + compile the LangGraph
  main.py           # CLI entry point (--verbose shows tool calls)
```

## Free-tier notes (football-data.org)

Verified against the live API on 2026-06-17:

- **World Cup is on the free tier.** `WC` (competition id 2000) has
  `plan: TIER_ONE`; the 2026 season runs 2026-06-11 → 2026-07-19. Every
  `WC` resource (matches, standings, teams, scorers) requires the token — calls
  without it return **403**, not public data.
- **Rate limit: 10 requests/minute.** Exceeding it returns **429**; a module-level
  **TTL cache** in the client dedupes repeated calls so parallel agents and rapid
  re-runs stay under it.
- **No per-player per-match stats.** Match `lineup`, `statistics`, `bench`, and
  `formation` fields are paid-tier only. `player_agent` reads the competition
  **scorers** endpoint (free) and enriches the player with a TheSportsDB bio.
- **Standings appear once the group stage has data.** Before then,
  `/competitions/WC/standings` can be empty — the matchup agent reports that
  rather than failing.

## Free-tier notes (Groq)

Groq rate-limits **per model**, on both tokens-per-minute (TPM) and
tokens-per-day (TPD). On the free tier `llama-3.3-70b-versatile` is ~12k TPM /
100k TPD, so a burst of agents all on 70b will rate-limit. This project keeps the
agent tool-loops on `llama-3.1-8b-instant` (separate, larger budget) and uses 70b
only for the final synthesis, which **auto-falls back to 8b** if the 70b daily
budget is exhausted. A 429 self-heals via a short backoff.

## News layer notes

- The RSS feeds are general football feeds; the news agent keeps only items that
  name the focus team. Coverage varies by team and how fresh the feed is. Add a
  free `TAVILY_API_KEY` for consistently team-specific, injury/lineup-level news.
- Every news call degrades gracefully: a dead feed, a Tavily error, or no matching
  headlines yields a clear "no recent news" section, never a crash.

## Future enhancements

- Next-opponent resolution + head-to-head history (TheSportsDB, free key `123`).
- Deep player profiles (TheSportsDB bios: position / age / club / nationality).
- Prediction / odds context (the-odds-api free tier).
