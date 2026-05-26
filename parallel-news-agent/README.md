# Parallel News Agent

A LangGraph map-reduce pipeline that researches multiple news topics simultaneously and assembles them into a single executive briefing.

## How it works

```
START → topic_dispatcher → news_analyst ×N (parallel)
                                ↓ (fan-in, all complete)
                         report_assembler → END
```

- **topic_dispatcher** — conditional edge that returns a `List[Send]`, launching one `news_analyst` per topic in the same superstep
- **news_analyst** — searches Tavily for recent news and writes a structured brief using Groq (llama-3.3-70b-versatile); runs in parallel, one instance per topic
- **report_assembler** — combines all briefs into a formatted daily intelligence report after all parallel agents finish

The `briefs` field in `OverallState` uses `Annotated[list[str], operator.add]` as a reducer so parallel writes from multiple agents are safely concatenated instead of overwriting each other.

## Stack

| Component | Tool |
|-----------|------|
| Orchestration | LangGraph |
| LLM | Groq (llama-3.3-70b-versatile) |
| Web search | Tavily |
| Language | Python 3.12+ |

## Setup

```bash
cd parallel-news-agent
uv sync
```

Create a `.env` file:

```
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## Run

```bash
uv run src/main.py
```

The agent searches 5 topics in parallel and writes the report to `report.md`.
