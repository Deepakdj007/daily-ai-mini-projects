# toolsmith

An agent that writes its own tools. It starts with two — a calculator and
today's date. You give it a task neither can do. It writes a Python function, a
guard reads the source and either refuses it with a reason or lets it through,
the function runs in a locked-down subprocess against its own tests and against
inputs it has never seen, and only then does it join the tool list —
mid-conversation, so the very next round can call it.

The tools it keeps are ordinary `.py` files in `generated/` that you can open
and read.

## Setup

```
uv sync
cp .env.example .env        # then paste a free key from console.groq.com/keys
```

## The two commands that are the whole point

```
PYTHONPATH=. uv run python -m src.main solve gstin_1 --fresh --show-source
PYTHONPATH=. uv run python -m src.main solve gstin_2
```

The first writes a tool and uses it. The second calls the tool it already has:

```
gstin_1     1,436 completion tokens   3 attempts, 1 registered   correct
gstin_2       269 completion tokens   reused the tool from gstin_1
isoweek_1   2,067 completion tokens   3 attempts, 1 registered   correct
isoweek_2     107 completion tokens   reused the tool from isoweek_1
```

The rejected attempts are the interesting part. On `isoweek_1` the first tool
passed both of the model's own tests and failed all three held-out checks,
because it stamped the label with the calendar year instead of the ISO
week-numbering year — and every held-out date sits near a year boundary, which
is exactly the case nobody puts in an example.

## Everything else

```
PYTHONPATH=. uv run python -m src.main redteam  # 19 hostile tool bodies, both layers
PYTHONPATH=. uv run python -m src.main tasks    # the four demo tasks
PYTHONPATH=. uv run python -m src.main tools    # what it can call right now
PYTHONPATH=. uv run python -m src.main inspect   # the tool it wrote
PYTHONPATH=. uv run python -m src.main clean    # back to two tools
```

`redteam` and `tasks` spend no tokens. Every module also checks itself — `uv
run python -m src.guard`, `-m src.sandbox`, `-m src.registry` and so on each
run their own assertions and print what they proved.

## About the sandbox

**Generated code runs as you, on your machine.** The guard refuses source it
cannot prove safe, and the sandbox builds the child's environment instead of
inheriting it — so `GROQ_API_KEY` is not in the process at all — pins the
working directory to a throwaway folder, installs an audit hook that cannot be
removed, denies every file write and every read outside the standard library,
and kills the whole process tree on timeout.

`src/redteam.py` reports exactly what that stops, including the table where the
guard is removed and the sandbox has to hold the line alone. Run it; the gap
between those two columns is the argument for having both layers.

It is still not a container and not a VM. There is no memory limit, no disk
quota and no privilege drop, and a spin loop costs you a core for ten seconds.
Treat it as a strong brake on a model that goes wrong, not as a wall around a
model that has been turned against you.
