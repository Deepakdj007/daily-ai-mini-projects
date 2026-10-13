# rag-debugger

A RAG debugger that shows why retrieval failed. It runs a plain Chroma top-k retriever, and on the side an LLM judge (Groq `openai/gpt-oss-120b`) reads a wider pool of 10 passages. The judge scores each one without seeing its rank. Every passage lands in one of four cells: HIT, NOISE, MISS or CUT. The pattern of cells names the failure: HEALTHY, NOISY_CONTEXT, RETRIEVAL_MISS or CORPUS_GAP. Each run is recorded as timed steps and saved as JSON in `output/traces/`. The data is a fictional home-textiles store, Kumkum Home.

```bash
uv sync
cp .env.example .env        # add a free key from https://console.groq.com/keys
PYTHONPATH=. uv run python -m src.main demo
PYTHONPATH=. uv run python -m src.main debug "Is there a warranty on brass diyas?" --top-k 3
```

The full walkthrough is in `build-rag-debugger-guide.md`.
