# slm-llm-router

A hybrid router that answers with a small local model, checks the answer with a
**local** verifier, and escalates to a large hosted model only when the check
fails. It ships with the measurement harness that decides whether the routing
was worth doing — including the controls that most "we cut costs 90%" posts
leave out.

Small tier: `qwen3.5:4b` on Ollama, $0 marginal API cost.
Large tier: `openai/gpt-oss-120b` on Groq, $0.15 in / $0.60 out per 1M tokens.

---

## The catch this project is built around

With a free local tier, `savings% = 1 − (fraction of queries sent to the large
model)`. So **a router that flips a weighted coin at p=0.1 reports "90% cheaper"
too.** The savings number is a property of the threshold, not of the router.

That is why the leaderboard carries three controls next to the cascade:

| row | what it tells you |
|---|---|
| **baseline** — everything to the LLM | the cost and quality ceiling |
| **all SLM** — everything to the small model | the floor. If this is near baseline, the eval set is too easy and every other row is noise |
| **random @ f** — a coin flip at the same routing rate | the chord. A real router must beat this |
| **oracle @ f** — escalate exactly what the SLM got wrong | the ceiling. How much room there was to work with |
| **cascade** | the actual system |

The deliverable is `output/deferral_curve.png`: accuracy against the fraction of
traffic sent to the big model. Random deferral is a straight line. A router that
earns its complexity bows above it. Moving the threshold slides a point *along*
the curve — it never lifts the curve off the chord, which is what makes the
chart hard to fake.

## Pre-registered criterion

Declared in `src/report.py` before the first full run:

> Supported iff the lower bound of the 95% CI on quality retention is ≥ 0.90
> while the LLM-call rate is ≤ 0.20.

**Result: NOT MET.** Retention came in at 0.950 with a 95% CI of 0.908–0.983 —
the retention half passes — but the cascade escalates 32% of traffic, over the
20% ceiling. The criterion stands as written; moving it after seeing the numbers
would be the exact thing pre-registration exists to prevent.

---

## Measured results

120 items, `qwen3.5:4b` local against `openai/gpt-oss-120b`.

| strategy | accuracy | 95% CI | cost | savings | LLM rate |
|---|---:|---:|---:|---:|---:|
| baseline (all LLM) | 99.2% | 98–100% | $0.00824 | 0% | 100% |
| all SLM | 73.3% | 65–81% | $0.00000 | 100% | 0% |
| random @ 32% | 81.7% | — | $0.00260 | 68% | 32% |
| oracle @ 32% | 99.2% | 98–100% | $0.00301 | 63% | 27% |
| **cascade** | **94.2%** | **90–98%** | **$0.00360** | **56%** | **32%** |

**The router works.** At the same 32% routing rate the cascade scores 94.2%
against random deferral's 81.7% — 12.5 points of accuracy bought purely by
deciding *which* queries to escalate (McNemar p=0.031 vs baseline). The verifier
catches 78% of the small model's actual errors at 66% precision.

**But it does not cut cost 90%, and nothing could.** Two hard limits:

1. `qwen3.5:4b` is wrong on **26.7%** of this traffic, so any router that holds
   baseline quality must escalate at least 26.7% of it.
2. The queries it escalates are the expensive ones — they cost **1.37×** the
   average, eating 36.5% of the bill.

So **perfect routing caps at 63.5% savings** on this mix. The measured 56% sits
close to that ceiling. The honest headline is not "90% cheaper" — it is *"56%
cheaper at 95% of baseline quality, against a hard ceiling of 63%."*

### When 90% *is* real

The ceiling is set by your traffic, not your router:

| traffic slice | SLM accuracy | must escalate | max savings | cascade saves |
|---|---:|---:|---:|---:|
| easy only | 96% | 4% | 95% | **97%** |
| easy + medium | 83% | 17% | 80% | 70% |
| full mix (60/25/15) | 73% | 27% | 63% | 56% |
| hard only | 17% | 83% | 12% | 16% |

A support bot answering FAQs really does save ~90%+. The same router on hard
analytical traffic saves ~16% and adds latency. Before adopting this pattern,
measure your own mix — that number, not the router, decides whether it pays.

### The latency the cost table hides

p50 latency: baseline **0.5s**, cascade **4.6s**. The free tier is roughly nine
times slower, because a 4B on local hardware loses to a hosted 120B that answers
in 0.08s of server time. You are trading the user's seconds for your dollars.

---

## Setup

```bash
uv sync
cp .env.example .env          # add your free key from https://console.groq.com/keys
ollama pull qwen3.5:4b        # 3.4 GB
```

## Run

```bash
# a fast, complete, tiny run - spans every difficulty bucket
PYTHONPATH=. uv run python -m src.main all --limit 12

# the full 120-item run
PYTHONPATH=. uv run python -m src.main all

# route a single query and watch the decision
PYTHONPATH=. uv run python -m src.main ask "What is the chemical symbol for iron?"
PYTHONPATH=. uv run python -m src.main ask "A tank fills at 12 litres per minute for 7 minutes, then 30 litres drain out. How many litres remain?"
```

The second one is worth running. The small model answers **66**, the correct
answer is **54**, and the verifier *accepts it* — all three self-consistency
samples agree on the same wrong number. That is the 22% of errors the verifier
misses, live: self-consistency catches a model that is guessing, not a model
that is confidently and systematically wrong. Any cascade you ship has this
failure mode, and the only way to know its size is to measure it.

Answers are cached in `output/cache.sqlite`, so re-runs are instant and free.
Every strategy and the whole strictness sweep replay that cache — only `run`
ever calls a model.

## Module smoke tests

```bash
PYTHONPATH=. uv run python -m src.grader     # asserts the substring-bug cases fail correctly
PYTHONPATH=. uv run python -m src.verifier   # the escalation checks
PYTHONPATH=. uv run python -m src.stats      # bootstrap CIs and McNemar
PYTHONPATH=. uv run python -m src.pricing    # cost ledger
PYTHONPATH=. uv run python -m src.calibrate  # is there a capability gap to route across?
```

Run `src.calibrate` first on any new model or dataset. If the small and large
tiers score the same on the hard bucket, there is nothing to route across and
every downstream number is noise.

---

## The eval set

120 items, graded by exact match — no LLM judge anywhere, which removes
self-preference bias entirely.

- **80 authored** production-traffic items: ticket routing, field extraction,
  unit conversion, date/format normalisation, one-step arithmetic.
- **40 vendored verbatim from GSM8K** (`data/LICENSE-gsm8k`, MIT, © 2021
  OpenAI). Difficulty comes from each problem's own step count, not our opinion.
  Vendoring the hard half means the questions the small model fails on were not
  written by the person reporting the result.

Mix is 60% easy / 25% medium / 15% hard, matching production traffic. That is
the thesis, not a thumb on the scale: **routing saves money because most real
queries never needed a frontier model.** Restrict the run to the hard bucket and
the savings collapse — if your traffic is all-hard, routing saves you nothing.

## If your number differs

It will. Expected outcomes by configuration:

| tier | `SLM_OLLAMA_MODEL` | download | expect |
|---|---|---|---|
| A (default) | `qwen3.5:4b` | 3.4 GB | best quality, lowest escalation |
| B | `qwen3.5:2b-q4_K_M` | 1.9 GB | weaker model escalates more, savings fall |
| C | `qwen3.5:0.8b` | 1.0 GB | smoke test only |
| D | `SLM_PROVIDER=groq` | none | whole project runs, savings drop to ~50% |

Tier D is fully supported — no Ollama required. Its ceiling is low for a
concrete reason: `gpt-oss-20b` is only **2× cheaper** than `gpt-oss-120b`, and
the price gap between your tiers is what sets the savings ceiling.

**Tier D also breaks the verifier's economics, measurably.** Run
`SLM_PROVIDER=groq ... --limit 12` and the default strictness-4 cascade reports
**−104% savings** — it costs more than twice the baseline. Self-consistency
calls the small model three times per query, and at only a 2× price gap those
three calls cost more than the single big-model call they were meant to avoid.

| strictness | Tier D savings | accuracy |
|---:|---:|---:|
| 1–3 (format + hedge only) | **+27%** | 100% |
| 4 (adds self-consistency, k=3) | **−104%** | 100% |

So the verifier's price is only free when the small tier is. On a paid small
tier, drop to `HEADLINE_STRICTNESS=3` and use the cheap checks. This is the
sharpest version of the whole lesson: **a cascade is only worth building when
the thing doing the verifying is far cheaper than the thing being avoided.**

- **A lower number than the guide's** usually means a weaker small model, so
  more escalation.
- **A higher number** is a warning, not a win: check the `all SLM` row. If it is
  close to baseline, your questions are too easy to distinguish the tiers.

## Two things the numbers will tell you that most write-ups skip

**Free is not fast.** On a CPU-only laptop the local 4B is often *slower* than a
hosted 120B, which answers in well under a second of server time. You are
trading the user's seconds for your dollars. That is a product decision, not an
engineering one, and the `p50 s` column makes it visible.

**Escalation is paid twice.** When the verifier defers, the small model's tokens
and latency are already spent. The cost column includes that double payment
rather than quietly dropping it.

## Layout

```
data/queries.json        120 items, content-hashed into every leaderboard
data/authored.json       the 80 hand-written items
data/LICENSE-gsm8k       MIT licence for the vendored slice
src/config.py            every knob, in one place
src/pricing.py           USD-per-token ledger
src/tiers.py             the two model callers behind one interface
src/cache.py             sqlite cache; key includes the prompt version
src/grader.py            word-boundary exact match
src/verifier.py          the local escalation gate
src/matrix.py            answer every item once (the only module that spends money)
src/strategies.py        cascade + the three controls, replayed offline
src/stats.py             bootstrap CIs, exact McNemar
src/evaluate.py          scoring
src/report.py            leaderboard, terminal + markdown
src/plot.py              the deferral curve
```

## Recommended: freeze the dataset

`output/leaderboard.md` stamps the dataset's sha256 on every run. Committing
`data/queries.json` before a run adds the git hash too, so a result can always
be traced to the exact questions that produced it.
