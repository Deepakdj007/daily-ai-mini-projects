# Context layer leaderboard

> Retrieval and summarisation earn their place iff, at the same 900-token budget, the full pipeline beats cap+pin+window by at least 20 points on the out-of-window needles, with exact-McNemar p < 0.01.

**Result: MET.** full 9/23 (39%, 95% CI [22%, 61%]) vs pin 1/23 (4%) = +35 points; discordant 8:0, exact McNemar p = 0.008. At n=23 one probe is 4.3 points, so the interval is wide and only a gap this large is resolvable.

`openai/gpt-oss-120b` | scenario `sha256:d809ff5652a5` | budget 900 tok | summary leakage 0/27 | cached tokens rate-limit exempt: False

## Validity gates

| check | result | detail |
| --- | --- | --- |
| no contamination | pass | 0 rows answered correctly from a context the grep proves the fact was absent from |
| recent zone is a clean sweep | pass | every arm answers every fact inside its own window |
| null context scores zero | pass | no planted fact is guessable without the conversation |
| no-pin control fails the pinned probe | pass | the pinned block really is removed when pin is off |
| tokenizer tracks the server | **FAIL** | local estimate vs Groq prompt_tokens: mean x1.237, worst divergence 60.5%. Every budget number, the headroom claim and the cost figure rest on this ratio being ~1. |
| truncation under ceiling | pass | 0.0% of rows lost their answer to the completion cap |

## The ladder

| arm | what it adds | out-of-window recall | recent | turns held | assembled tok |
| --- | --- | ---: | ---: | ---: | ---: |
| none | null context | 0/23 (0%) | 0/4 | 0 | 129 |
| raw | raw history | 0/3 (0%) | 0/0 | 100 | 16,319 |
| window | + window | 0/23 (0%) | 4/4 | 4 | 772 |
| pin | + pin | 1/23 (4%) | 4/4 | 7 | 781 |
| full | + retrieve | 9/23 (39%) | 4/4 | 5 | 895 |
| no-pin | full - pin | 0/0 | 0/0 | 5 | 857 |

## Did the layer fail, or the model?

| arm | present & correct | present & missed | absent & missed | absent & correct |
| --- | ---: | ---: | ---: | ---: |
| none | 0 | 0 | 23 | 0 |
| raw | 0 | 0 | 0 | 0 |
| window | 0 | 0 | 23 | 0 |
| pin | 1 | 0 | 22 | 0 |
| full | 9 | 1 | 13 | 0 |
| no-pin | 0 | 0 | 0 | 0 |

`absent & missed` is the layer's failure. `present & missed` is the model's: the fact was in the prompt and it still did not answer.

## What the run found that the design did not predict

| arm | out-of-window recall, fact in prose | fact in tool output |
| --- | ---: | ---: |
| none | 0/8 | 0/15 |
| raw | 0/1 | 0/2 |
| window | 0/8 | 0/15 |
| pin | 0/8 | 1/15 |
| full | 0/8 | 9/15 |
| no-pin | - | - |

BM25 matches tokens. A value sitting in a JSON field with a distinctive name is findable; the same value paraphrased into a sentence is not. The build gate that forces low lexical overlap between a probe and the transcript - the gate that stops the retrieval test being rigged - is precisely what makes prose invisible to a lexical retriever. Swapping BM25 for embeddings is the obvious next move, and this table is the reason.

