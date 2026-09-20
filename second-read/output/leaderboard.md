# second-read leaderboard

Subject `medgemma1.5:4b` (`433252621ab154668b5`), profile `full`, probes `sha256:84d5451ee54499cac86`, temperature 0.0, top_k 1, num_ctx 8192, images 896px.

## Validity gates

| gate | ok | detail |
|---|:--:|---|
| blind sits at the 50% construction floor | yes | 50/100 = 50%, 95% CI [40%, 60%] |
| blind recovers no minted lab value under its own name | yes | 0 of 269 values over 20 pages (2 loose numeric collisions in blind prose, expected and not a leak) |
| image canary: read sighted, silent blind | yes | sighted 16/16, blind 0/16 |
| image bytes: >0 on every sighted row, 0 on every blind row | yes | clean |
| scored rows completed (no truncation, no unparsed answer) | yes | 0/100 incomplete, excluded |
| repeat agrees with sighted (else VOID) | yes | 0/100 disagreements |
| balance survived attrition | yes | 100 balanced of 100 complete |
| probe set stamped | yes | sha256:84d5451ee54499cac86 |

## Accuracy by condition

| model | n | sighted | blind | shuffled | repeat agrees | flagged |
|---|--:|--:|--:|--:|--:|--:|
| medgemma1.5:4b | 100 | 75% | 50% | 44% | 100% | 57% |
| gemma3:4b | 100 | 58% | 51% | 54% | 100% | 77% |

## Outcome cells

| model | EARNED | LUCKY | PRIOR_ERROR | MISREAD | IMAGE_SHAPED | UNSTABLE | INCOMPLETE |
|---|--:|--:|--:|--:|--:|--:|--:|
| medgemma1.5:4b | 24 | 41 | 16 | 5 | 14 | 0 | 0 |
| gemma3:4b | 6 | 43 | 34 | 5 | 12 | 0 | 0 |

EARNED = correct and grounded. LUCKY = correct but said with the image removed. PRIOR_ERROR = wrong and said with the image removed. MISREAD = wrong although the image drove the answer - the audit cannot catch this cell by construction.

## Pre-registered criterion

> The ablation audit earns its place iff, on the authored IU-Xray probes where every condition completed, the claims it flags as language prior - the model gives the same answer with the image deleted - are correct at least 15 points less often than the claims it clears, with Fisher exact p < 0.05, while three bounds hold: blind accuracy sits within 5 points of the 50% floor the probes are balanced to; blind recovery of the minted lab values is exactly zero; and the image canary reads at least 90% sighted and 0% blind. The flag rate itself, the gemma3:4b twin and the unmentioned-as-negative shadow set are reported and are not part of pass or fail.

- 57/100 claims flagged as language prior (57%)
- flagged claims correct 72% (95% CI [59%, 82%]); cleared claims correct 79% (95% CI [65%, 89%])
- gap +7 points (95% CI -10 to +24); need >= 15
- Fisher exact p = 0.4883; need < 0.05
- bound - blind sits at the 50% construction floor: holds
- bound - blind recovers no minted lab value under its own name: holds
- bound - image canary: read sighted, silent blind: holds

**flagged claims are +7 points less accurate than cleared ones, p = 0.4883**

**Result: NOT MET.**

The criterion sentence is unchanged from before the run. One instrument behind it was corrected afterwards: the lab bound originally counted a minted value as recovered if the number appeared anywhere in the blind answer, which scored two coincidences in a six-thousand-character essay as leaks. It now requires the value to arrive under its own analyte name. That correction moves a bound, not the headline - the gap clause fails on its own numbers either way. See FINDINGS.md #13.

## Secondary, reported not scored

- sighted: medgemma1.5 75% vs gemma3:4b 58% on 100 shared probes: +17 points (95% CI +2 to +32), discordant 39:22, exact McNemar p = 0.0396
- language-prior rate: medgemma1.5 57% vs gemma3:4b 77% - the difference is what medical tuning installed as prior
- lab extraction: mean field accuracy 61% over 20 synthetic pages (exact match on eight fields; a ceiling, not a benchmark)
- shadow set (unmentioned counted as no): sighted 86% 'correct' on 35 probes - the number the shortcut would have published

## What each condition is for

- `sighted` (the product's own answer) predicted: medgemma1.5 above 70% on balanced probes; the twin nearer 55%
- `blind` (does the claim survive deleting the evidence) predicted: at the 50% floor for both models, with medgemma1.5 agreeing with its sighted self more often
- `shuffled` (does it survive replacing the evidence) predicted: between blind and sighted; the gap to sighted is what the specific film buys
- `repeat` (the noise floor at temperature zero) predicted: under 5% disagreement - greedy decoding on a fixed prompt should be near-deterministic
