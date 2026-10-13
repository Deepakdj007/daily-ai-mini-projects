# crag-railways

A Corrective RAG (CRAG) help desk for Indian Railways questions. Plain RAG trusts whatever the retriever brings back.
CRAG grades the passages first, then picks one action. **CORRECT** keeps only the sentences that answer the
question. **INCORRECT** rewrites the question and searches the web. **AMBIGUOUS** does both. The flow is a
LangGraph `StateGraph` running on Groq `openai/gpt-oss-120b`, with an in-memory Chroma KB and keyless DuckDuckGo search.

The KB passages are simplified demo notes, not an official IRCTC source.

```bash
uv sync
cp .env.example .env            # add GROQ_API_KEY (free: https://console.groq.com/keys)
PYTHONPATH=. uv run python -m src.main demo                     # 3 questions, plain RAG vs CRAG
PYTHONPATH=. uv run python -m src.main ask "Can I change my boarding station?"
PYTHONPATH=. uv run python -m src.main ask "..." --plain        # no evaluator, for contrast
```
