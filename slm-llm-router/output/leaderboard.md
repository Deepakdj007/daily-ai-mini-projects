# SLM+LLM router leaderboard

- items: **120**
- dataset: `sha256:2dc0d8a3f06b git:UNCOMMITTED`
- small tier: `qwen3.5:4b (local)`
- large tier: `openai/gpt-oss-120b`
- headline operating point: verifier strictness **4**

## Pre-registered criterion

> Supported iff the lower bound of the 95% CI on quality retention is >= 0.90 while the LLM-call rate is <= 0.20.

**NOT MET** — retention 0.950 (95% CI 0.908-0.983), LLM-call rate 32%

## Strategies

| strategy | accuracy | 95% CI | cost (USD) | savings | LLM rate | retention | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline (LLM) | 99.2% | 98%–100% | $0.00824 | 0.0% | 100% | 1.000 | 1.000 |
| all SLM | 73.3% | 65%–81% | $0.00000 | 100.0% | 0% | 0.739 | 0.000 |
| random@32% | 81.7% | — | $0.00260 | 68.4% | 32% | 0.824 | — |
| oracle@32% | 99.2% | 98%–100% | $0.00301 | 63.5% | 27% | 1.000 | 1.000 |
| cascade s4 k3 | 94.2% | 90%–98% | $0.00360 | 56.4% | 32% | 0.950 | 0.031 |

The random row is an expectation over 1,000 draws, not a realised per-item vector, so a confidence interval and a paired p-value are not defined for it.

## Deferral sweep

| strictness | LLM rate | accuracy | cost (USD) | verifier precision | verifier recall |
|---:|---:|---:|---:|---:|---:|
| 0 | 0% | 73.3% | $0.00000 | 0% | 0% |
| 1 | 0% | 73.3% | $0.00000 | 0% | 0% |
| 2 | 1% | 74.2% | $0.00013 | 100% | 3% |
| 3 | 1% | 74.2% | $0.00013 | 100% | 3% |
| 4 | 32% | 94.2% | $0.00360 | 66% | 78% |
| 5 | 100% | 99.2% | $0.00824 | 27% | 100% |

## What sets the savings ceiling

A router that holds baseline quality has to escalate at least as often as the small model is wrong — and the items it escalates are the expensive ones, so they eat a bigger share of the bill than of the traffic. `max savings` is the most that perfect routing could save **while still matching baseline accuracy**. It is a property of your traffic and your small model, not of your router.

The cascade can exceed `max savings` on a slice, as it does on easy-only. That is not a better router — it is a router escalating less than it should and buying the difference in accuracy.

| traffic slice | n | SLM accuracy | must escalate | max savings | cascade savings | cascade accuracy |
|---|---:|---:|---:|---:|---:|---:|
| easy only | 72 | 96% | 4% | 95% | 97% | 97% |
| easy + medium | 102 | 83% | 17% | 80% | 70% | 96% |
| full mix (60/25/15) | 120 | 73% | 27% | 63% | 56% | 94% |
| hard only | 18 | 17% | 83% | 12% | 16% | 83% |

## Accuracy by difficulty

| bucket | n | SLM | LLM | gap |
|---|---:|---:|---:|---:|
| easy | 72 | 96% | 99% | +3% |
| medium | 30 | 53% | 100% | +47% |
| hard | 18 | 17% | 100% | +83% |

## Reading this table

`savings` is not a property of the router. With a local small tier at $0, savings is `1 - LLM-call-rate`, so the random control reports the same savings as the cascade at the same rate. What the cascade has to earn is the **accuracy** at that rate — the gap between its row and the random row, bounded above by the oracle row.
