# smolagents Code Agent

A [smolagents](https://github.com/huggingface/smolagents) `CodeAgent` powered by Groq's free `llama-3.3-70b-versatile` that solves any problem by **writing and running Python**. It can search the web (DuckDuckGo), call a custom `prime_factors` tool, and import pandas/numpy/math/statistics in its generated code, looping until it has an answer.

## Run it

```bash
uv sync
cp .env.example .env   # then paste your key from https://console.groq.com/keys
PYTHONPATH=. uv run python -m src.main
```

Type a problem at the `Problem>` prompt and watch the agent write code, run it, and answer. Try:

- `What's the 118th Fibonacci number?`
- `Given [8,3,5,1,9,2], give the mean, median, and standard deviation.`
- `Who is the current CEO of Hugging Face?`
