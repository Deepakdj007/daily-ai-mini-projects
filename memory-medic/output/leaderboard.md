# memory medic - silent decay leaderboard

> The hygiene sweep earns its place iff, on the source-drift probes, `sweep` beats `freshness` by at least 25 points with exact McNemar p < 0.01, while passing at least 90% of the stable-fact controls and losing nothing on the probes `freshness` already handled.

**MET** - sweep 10/10 vs freshness 0/10 on source drift: +100% (95% CI +100% to +100%), exact McNemar p=0.001953 [10 vs 0 discordant]; stable controls 100%; regression elsewhere 0


Run 2026-09-05 · answers `openai/gpt-oss-120b` · extraction `openai/gpt-oss-20b` · fixture `sha256:c513472e4d6e` · probes asked as of 2026-11-20 · 18,256 tokens.

## Validity gates

| gate | result | detail |
| --- | --- | --- |
| no context leaks | PASS | 0 rows where an arm saw a value it should not have |
| null arm cannot guess | PASS | 0/22 planted probes passed with no memory |
| controls answerable in every arm | PASS | worst arm passes 67% of the in-message controls |
| no confabulation on unknowns | PASS | 0/15 answered a question never discussed |
| no failed model calls | PASS | 0 calls failed |
| fixture is stamped | PASS | sha256:c513472e4d6e |

## Probes passed, by arm and category

| arm | aged | announced | in message | negative | scheduled | scope pair | source drift | stable control |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `none` | 0/8 | 0/4 | 2/3 | 3/3 | 4/4 | 0/4 | 0/10 | 0/4 |
| `overwrite` | 0/8 | 2/4 | 3/3 | 3/3 | 3/4 | 4/4 | 0/10 | 4/4 |
| `bitemporal` | 0/8 | 2/4 | 3/3 | 3/3 | 3/4 | 4/4 | 0/10 | 4/4 |
| `freshness` | 7/8 | 4/4 | 2/3 | 3/3 | 3/4 | 4/4 | 0/10 | 4/4 |
| `sweep` | 7/8 | 4/4 | 2/3 | 3/3 | 3/4 | 4/4 | 10/10 | 4/4 |

## What each rung adds

| arm | switch | why it is here |
| --- | --- | --- |
| `none` | - | Guessability floor. Any probe this arm passes was answerable without remembering anything, and cannot be evidence for a memory mechanism. |
| `overwrite` | resolves | The incumbent. A contradiction deletes the old row at write time, which is what a published memory library does by default. |
| `bitemporal` | history | Supersede instead of delete, and filter reads by valid time. The store a careful engineer would ship. |
| `freshness` | freshness | The cheap half of staleness: say how old a memory is at read time and let the model hedge. Costs nothing and needs no background work. |
| `sweep` | sweep | Headline. Durable state changes between turns, and the only rung that re-reads a source nobody mentioned. |
