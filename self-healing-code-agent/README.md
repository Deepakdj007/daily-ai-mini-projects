# Self-Healing Code Agent

A code agent that writes Python, runs it, reads its own tracebacks, and fixes its
own bugs — looping until the `assert`-based tests it wrote for itself all pass.
Built on [Pydantic AI](https://ai.pydantic.dev/) with Groq's free
`llama-3.3-70b-versatile`. The agent calls a `run_python` tool that executes each
candidate script in an isolated subprocess; on failure the traceback is handed
straight back to the model so it can correct the code and retry.

## Run it

```bash
cp .env.example .env          # then paste your key from https://console.groq.com/keys
PYTHONPATH=. uv run python -m src.main
```

Give it a bug-prone task, e.g. *"Write median(nums) that returns the median of a
list; handle even-length lists; test it on [1,2,3,4] and [5,1,3]."* Watch it fail,
read the error, and fix itself.

> Note: generated code runs in a subprocess on your machine. It is isolated from
> crashes and infinite loops (10s timeout) but is **not** a security sandbox — run
> only tasks you trust.
