# jev-switchboard

A support desk for a small Indian home-decor store. Jev (TypeSafe's System One model) reads each customer message and sends it to one of five places: a database lookup, a small LLM, a big LLM, a human, or a clarifying question. Jev also checks every LLM reply before it goes out. The routing itself is plain `if` statements on Jev's typed answers and confidence scores.

```bash
uv sync
cp .env.example .env   # add TYPESAFE_API_KEY and GROQ_API_KEY
PYTHONPATH=. uv run python -m src.desk ask "Where is my order KH-1042?"
PYTHONPATH=. uv run python -m src.desk demo
PYTHONPATH=. uv run python -m src.desk race
```
