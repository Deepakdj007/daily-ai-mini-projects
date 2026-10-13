# semantic-cache

A support agent that checks a semantic cache before it calls the LLM. Each question is embedded locally (Chroma's default all-MiniLM-L6-v2). If an earlier question means the same thing (cosine similarity at or above the threshold), the cached answer is returned in milliseconds and costs nothing. `replay` runs 60 customer questions with and without the cache and saves `cache_report.png` (cumulative cost + latency). `sweep` shows the hits and false hits at every threshold, and `invalidate-demo` serves a stale answer, then removes it three ways (targeted delete, version bump, TTL). Gemini `gemini-3.5-flash-lite` on the free tier.

```bash
uv sync
cp .env.example .env            # add GEMINI_API_KEY from https://aistudio.google.com/apikey
PYTHONPATH=. uv run python -m src.main sweep
PYTHONPATH=. uv run python -m src.main replay
PYTHONPATH=. uv run python -m src.main invalidate-demo
PYTHONPATH=. uv run python -m src.main ask "What is the return window?"
```
