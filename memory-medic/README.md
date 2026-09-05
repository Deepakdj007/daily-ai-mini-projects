# memory-medic

An agent that notices its own memories have gone stale, and repairs them — with
the ablation that says which parts of that machinery actually earn their place.

Verified 2026-09-05 — langgraph 1.2.11, langgraph-checkpoint-sqlite 3.1.1,
sqlite-vec 0.1.9, sentence-transformers 6.0.1, groq 1.7.0, streamlit 1.61.1,
Python 3.12, Windows. Free Groq `openai/gpt-oss-120b` (answers) and
`openai/gpt-oss-20b` (extraction), local MiniLM embeddings.

---

## The catch this project is built around

A memory that was *never* true is easy. It contradicts something, retrieval
scores it badly, a user corrects it.

A memory that **was** true is the dangerous one. "Works at Zeta Retail" was
right for two years. It embeds perfectly, retrieves with the highest score in
the store, and the model states it with complete confidence. Nothing about it
looks wrong, because until recently nothing about it *was* wrong.

There are two ways this happens, and almost every tutorial handles only the
first:

1. **Somebody says otherwise.** "I moved to Bengaluru." Mem0-style resolvers
   handle this by deleting the old row at write time.
2. **Nobody says anything.** The fact simply outlived its shelf life, or the
   document it came from was edited while no one was talking about it. There is
   no message to trigger on, no contradiction to detect, and no benchmark that
   measures it.

This project does the second, and measures whether it was worth doing.

## The five mechanisms

| layer | what it does | file |
|---|---|---|
| bitemporal store | two time axes: when a fact was true, and when we believed it. A contradiction closes the old window and keeps the row | `src/store.py` |
| scope-aware writes | facts qualified by different contexts never meet the classifier, so a preference that holds at work cannot delete one that holds at home | `src/classify.py` |
| freshness at read time | every recalled memory arrives carrying its age, and the model is told it may hedge | `src/recall.py` |
| the hygiene sweep | four detectors that find rot with nobody prompting: aged past its half-life, source changed, two live facts on one key, window closed | `src/detect.py` |
| the human gate | repairs the agent is not confident about park on a durable `interrupt()` and wait in a Streamlit inbox | `src/graph.py` |

Facts carry a **volatility class** that decides their shelf life: `stable`
(3650 days), `slow` (180), `fast` (30), and `scheduled`, which does not decay at
all because it ends on a date instead.

## What the run found

Five arms, forty probes, one switch between neighbouring rungs, checked at
import by `assert_single_switch()`.

| arm | aged | announced | scheduled | scope pair | **source drift** | stable control |
|---|---|---|---|---|---|---|
| `none` | 0/8 | 0/4 | 4/4 | 0/4 | **0/10** | 0/4 |
| `overwrite` | 0/8 | 2/4 | 3/4 | 4/4 | **0/10** | 4/4 |
| `bitemporal` | 0/8 | 2/4 | 3/4 | 4/4 | **0/10** | 4/4 |
| `freshness` | 7/8 | 4/4 | 3/4 | 4/4 | **0/10** | 4/4 |
| `sweep` | 7/8 | 4/4 | 3/4 | 4/4 | **10/10** | 4/4 |

Pre-registered criterion, declared in `src/report.py` before the run:

> The hygiene sweep earns its place iff, on the source-drift probes, `sweep`
> beats `freshness` by at least 25 points with exact McNemar p < 0.01, while
> passing at least 90% of the stable-fact controls and losing nothing on the
> probes `freshness` already handled.

**MET** — 10/10 versus 0/10, +100% (95% CI +100% to +100%), exact McNemar
p = 0.00195 on 10 discordant pairs, stable controls 100%, no regression
elsewhere. All six validity gates pass, including the null arm scoring 0/22 on
planted probes.

### Three things the numbers say that the design did not predict

**Keeping history changed no answer at all.** `overwrite` and `bitemporal` are
identical on every category. Supersede-not-delete costs nothing and buys nothing
*for answering questions about the present*; what it buys is the ability to
answer about the past, which the demo shows and no probe here asks. It is an
auditability property, not an accuracy one, and the table should not be read as
saying otherwise.

**The cheapest rung did most of the work.** Simply rendering how old a memory is
took the aged probes from 0/8 to 7/8 and the announced ones from 2/4 to 4/4.
Before that flag, the model asserted five-month-old facts as current in 7 of 8
cases. That is one line of prompt and one field, and it beats anything else on
the ladder per unit of effort.

**The sweep's contribution is real but narrow.** It is the only rung that moves
source drift, and it moves it completely. That is precisely the case nothing
else can see — nobody mentioned the change, so there is nothing to trigger on
except going to look.

## Setup

```bash
uv sync
cp .env.example .env          # then paste a free key from console.groq.com/keys
```

## Run it

```bash
# the whole story, a year of decay in about a minute
bash demo.sh                      # PowerShell: .\demo.ps1

# or step by step
uv run python -m src.main --as-of 2026-06-01 seed --reset
uv run python -m src.main --as-of 2026-06-10 say "I moved to Bengaluru last week."
uv run python -m src.main --as-of 2026-08-01 sweep --once
uv run python -m src.main --as-of 2026-08-03 timeline demo city

# the review inbox. CLOCK_AT is not decoration: the demo data is written
# against a frozen timeline, so without it the ages are measured from today.
CLOCK_AT=2026-08-03 uv run streamlit run src/inbox.py

# the measurement
uv run python -m src.main run --profile lite --yes
uv run python -m src.main report          # free, rebuilds from the results file
```

On Windows PowerShell an environment variable is set separately, in the same
shell: `$env:CLOCK_AT="2026-08-03"` and then the `uv run` line.

Every module is runnable on its own and asserts something real; several need no
API key at all:

```bash
uv run python -m src.store      # supersede keeps history, as_of picks by date
uv run python -m src.matcher    # the substring bug that inflates every arm
uv run python -m src.stats      # 8 one-way disagreements is the p<0.01 threshold
uv run python -m src.ladder     # one switch per rung
uv run python -m src.fixture    # the probe independence gates
uv run python -m src.main check # nothing reads the wall clock behind clock.py
```

## The clock

Everything takes its time from an injected clock, and `main.py check` greps for
anyone cheating. That is not tidiness: an evaluation of decay that cannot move
the clock cannot test decay. `--as-of 2026-12-01` runs the whole agent on that
date, which is how a year of rot fits in a minute.

## Files

```
src/
  config.py   clock.py    cache.py    llm.py      embed.py     # infrastructure
  store.py    recall.py                                        # the bitemporal store
  extract.py  classify.py policy.py   repair.py                # the write path
  detect.py   propose.py  sweep.py    graph.py    nodes.py     # the hygiene sweep
  turn.py     seed.py     main.py     inbox.py                 # the agent and its faces
  fixture.py  score.py    matcher.py  ladder.py                # the experiment
  harness.py  stats.py    report.py                            # running it, reading it
```

## Where this would fail

The sweep re-reads sources it already knows about. A fact whose source is a
conversation has nothing to re-read, so it can only ever age out — the store
cannot tell "still true" from "nobody has mentioned it in a while".

Half-lives are per class, not per person. Somebody who changes jobs every eight
months and somebody who has been at the same desk for a decade get the same
180-day slow half-life, and both are wrong.

The contradiction detector's semantic pass embeds every live scoped fact on
every sweep. That is fine at a few hundred memories and is not fine at a
million; it wants an index of what has changed since the last pass.

And the gate depends on somebody actually opening the inbox. A parked repair is
a memory that is knowingly wrong and staying that way until a human turns up.

`FINDINGS.md` records fourteen things this build got wrong on the way here,
including two probes that were measuring nothing and a gate that silently
stopped gating.
