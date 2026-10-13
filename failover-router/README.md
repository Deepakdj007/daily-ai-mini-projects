# failover-router

One `complete(messages)` call in front of three LLM providers: Gemini first, Groq as fallback, a local Ollama model as last resort. A support agent triages six tickets in one run. When a provider rate-limits, errors or has a bad key, the router retries with backoff, opens a circuit breaker and fails over mid-run. The customer-facing replies look the same either way. If everything is down, the run still finishes with a holding reply and flags the ticket for a human. `src/litellm_router.py` does the same job with LiteLLM's Router for comparison.

```bash
uv sync
cp .env.example .env            # add GEMINI_API_KEY and GROQ_API_KEY
ollama pull llama3.2:3b
PYTHONPATH=. uv run python -m src.main selftest                 # free, no API calls
PYTHONPATH=. uv run python -m src.main run                      # everything healthy
PYTHONPATH=. uv run python -m src.main run --kill gemini@3-4    # outage mid-run
PYTHONPATH=. uv run python -m src.main litellm --kill gemini@3-4
```
