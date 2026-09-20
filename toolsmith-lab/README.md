# toolsmith

An agent that writes its own tools. When it meets a task no tool of its can do,
it writes a Python function, a static guard reads the source and either refuses
it with a reason or lets it through, the function runs in a stripped-down
subprocess against its own tests and against inputs it has never seen, and only
then does it join the tool list — mid-conversation, so the next round can call
it. The tools it keeps are ordinary `.py` files in `generated/` that you can
open and read.

The interesting question is not whether it can do that. It is whether keeping
the tool is worth anything, so the project also measures that against a control
that writes the same code and throws it away every time. The criterion was
written before the run and it can come out NOT MET.

## Setup

```
uv sync
cp .env.example .env        # then paste a free key from console.groq.com/keys
```

## The part that costs nothing

```
PYTHONPATH=. uv run python -m tools.selftest        # every check that spends no tokens
PYTHONPATH=. uv run python -m src.main gates        # the corpus cannot be shortcut
PYTHONPATH=. uv run python -m src.main redteam      # 34 hostile tool bodies, both layers
PYTHONPATH=. uv run python -m src.main ladder       # the arms and what each was predicted to do
PYTHONPATH=. uv run python -m src.main tasks        # 24 tasks, 48 held-out inputs
```

## Watching it write a tool

```
PYTHONPATH=. uv run python -m src.main solve gstin_1 --arm persist --show-source --fresh
PYTHONPATH=. uv run python -m src.main inspect gstin_check
PYTHONPATH=. uv run python -m src.main solve gstin_2 --arm persist     # calls what it wrote
PYTHONPATH=. uv run python -m src.main solve gstin_2 --arm holdout     # writes it again
```

Those last two, back to back, are the whole argument.

## The measurement

```
PYTHONPATH=. uv run python -m src.main run --profile smoke    # ~40k tokens, one sitting
PYTHONPATH=. uv run python -m src.main run --profile lite     # ~2 days of the free tier
PYTHONPATH=. uv run python -m src.main run --profile full     # ~5 days
PYTHONPATH=. uv run python -m src.main report
PYTHONPATH=. uv run python -m src.main budget
```

The run is resumable. It stops itself at 95% of the daily allowance and the
same command picks up where it left off tomorrow.

## About the sandbox

**Generated code runs as you, on your machine.** The guard refuses source it
cannot prove safe and the sandbox builds the child's environment instead of
inheriting it — so `GROQ_API_KEY` is not in the process — pins the working
directory to a throwaway folder, installs an audit hook that cannot be removed,
denies every file write and every read outside the standard library, and kills
the whole process tree on timeout. `src/redteam.py` reports exactly what that
stops, including the table where the guard is removed and the sandbox has to
hold the line alone.

It is still not a container and not a VM. There is no memory limit, no disk
quota and no privilege drop, and a spin loop costs you a core for ten seconds.
Treat it as a strong brake on a model that goes wrong, not as a wall around a
model that has been turned against you.
