# hardened-rag

A retrieval pipeline measured against a knowledge base that has been poisoned,
and a ladder of seven defences that says which of them actually does anything.

## The catch this project is built around

A poisoned passage is not a low-quality passage. It is written to be the most
relevant thing in your index: it opens with the question and then answers it
wrongly. Every instinct a RAG tutorial gives you makes that worse. Better
embeddings rank it higher. A reranker promotes it. A longer top-k retrieves
more copies of it.

The published defence for this is isolate-then-aggregate: read each passage
alone so none of them can talk about the others, then take the majority answer.
That works on open-domain question answering, where a dozen web pages
independently know when the Battle of Hastings was.

A company knowledge base is the opposite shape. Each fact lives in exactly one
authoritative document. So the vote is one true passage against however many
copies the attacker uploaded, and the attacker chooses that number.

## The seven rungs

Each rung adds exactly one mechanism, and `src/ladder.py` refuses to import if
any rung differs from its parent by more than one switch.

| rung | adds | what it is for |
|---|---|---|
| `none` | nothing | the guessability floor |
| `naive` | dense top-5 in one prompt | the tutorial pipeline |
| `rerank` | a cross-encoder | the first thing anyone tries |
| `guard` | Prompt Guard 2 per passage | catches text that gives orders |
| `echo` | question-echo screen | catches text that quotes the question |
| `isolate` | one call per passage, claims verified | nothing can talk about anything else |
| `provenance` | conflicts settled by source tier and date | **headline** |

Plus one control, `majority`, which runs the textbook vote over the identical
claim table and loses.

The headline comparison is `provenance` against `isolate`, not against `naive`.
Beating the naive pipeline on safety is trivial: isolation already refuses to
assert the attacker's value. The open question is whether a **correct** answer
can be recovered from a poisoned context without becoming foolable again.

## What the model never sees

Passages reach the prompt as bodies only, numbered `[1]` to `[5]`, in an order
derived from a hash of the passage id. No tier, no date, no document title, no
filename. A build gate checks that none of those strings ever appears in a
rendered prompt.

That is the entire argument for the last rung. The tier and the date are read
by twenty lines of Python in `src/resolve.py`, and a passage cannot argue with
them, plead with them, or claim to supersede them.

## The attack matrix

Twelve conditions over the same questions, applied by metadata filter against
one shared index, so a poisoned passage has to earn its rank rather than being
handed a slot.

- `clean`, `absent` (the answer is withheld; abstaining is the correct output)
- `poison-p` (paraphrased) and `poison-v` (verbatim), at doses of 1, 3 and 5
- `inject-overt` and `inject-policy`
- `stale`: a real older version of the real document
- `faq`: the false-positive control for the echo screen
- `saturate`, `sametier`, `embedded`: three limits where the headline mechanism
  is expected to fail, reported rather than omitted

## What the run found

Day one of the free-tier budget: 691 of 1,776 rows, with the headline condition
and the clean bound both complete at 38 of 38 questions. All seven validity
gates pass.

| arm | clean | poisoned (N=3) |
|---|---:|---:|
| `none` | 0/38 | 0/38 |
| `naive` | 36/38 | 5/38, asserted the attacker's figure **31 times** |
| `rerank` | 38/38 | 5/38, asserted it 28 times |
| `guard` | 38/38 | 4/38, asserted it 29 times |
| `echo` | 38/38 | 4/38, asserted it 29 times |
| `isolate` | 23/38 | 4/38, asserted it **0 times** |
| `majority` | 23/38 | 4/38, asserted it **34 times** |
| `provenance` | 34/38 | **35/38**, asserted it 0 times |

Read three rows. `naive` is fooled 31 times out of 38, which is PoisonedRAG
reproduced on a synthetic corpus. `majority` is fooled **34** times, more often
than the naive pipeline it was supposed to improve on, because the vote is the
one mechanism the attacker controls the input to. `provenance` gets 35 right
and is never fooled once.

The two screens contributed almost nothing here, and the table says so: the
injection classifier quarantined 3 passages across 38 poisoned questions, and
the echo screen quarantined none. The reordered attack is not an instruction
and does not quote the question, so neither screen has anything to see.

`isolate` costs 15 correct answers on clean retrieval, dropping from 38 to 23.
That is the real price of reading passages one at a time, and it is why the
headline is stated against `isolate` rather than against `naive`.

**Headline:** on the 38 poisoned questions where an attack passage reached the
reader, `provenance` answered correctly 92% of the time against `isolate`'s
11%. That is +82 points, 95% CI +68 to +92, with 31 discordant pairs to zero
and exact McNemar p < 0.00001.

**Result: NOT MET.** The criterion carries three clauses and the headline is
not the one that failed. On the `absent` condition, where the answer has been
deleted from the corpus, it requires abstention in 80% of cases and
`provenance` abstains in 40%. It is not answering wrongly - it hedges in
another 40%, offering a forum post's figure with an explicit warning attached.
The project's own scoring function counts that as correct behaviour and reports
8 of 10; the criterion counts only silence and reports 4 of 10. Neither was
changed after the fact. See finding 7.

## Setup

```bash
uv sync
cp .env.example .env      # then paste a free key from https://console.groq.com/keys
```

## Run it

```bash
PYTHONPATH=. uv run python -m src.main build      # embed the corpus and the attacks
PYTHONPATH=. uv run python -m src.main gates      # every check that costs nothing
PYTHONPATH=. uv run python -m src.main ladder     # the rungs and their predictions

# one question, the whole audit trail - start here
PYTHONPATH=. uv run python -m src.main ask q01 --arm naive      --cond poison-p
PYTHONPATH=. uv run python -m src.main ask q01 --arm provenance --cond poison-p

PYTHONPATH=. uv run python -m src.main run --profile lite       # one day of free tier
PYTHONPATH=. uv run python -m src.main report
```

`bash demo.sh` runs the whole story end to end.

## The budget

Groq's free tier gives 200,000 tokens a day **per model**, and the full matrix
needs about three days of that. The run is resumable: rows already in
`output/results-<model>.json` are skipped, and the harness stops itself at 95%
of the daily ceiling rather than discovering it as a 429. Day one covers the
headline condition and the clean bound it is judged against, so an interrupted
run can still answer the question the project is about.

Prompt Guard has its own, much larger bucket, so every passage is scored once at
index time and the `guard` rung costs the run nothing.

`--profile lite` is ten questions and six arms, and fits in a single day.

## Files

- `data/paysetu.json` - the corpus, authored by `tools/author_corpus.py`
- `src/corpus.py`, `src/adversary.py` - passages, questions, and every attack
- `src/index.py`, `src/rerank.py`, `src/screen.py` - retrieval and the two screens
- `src/isolate.py`, `src/resolve.py` - per-passage claims, and the three resolvers
- `src/ladder.py`, `src/pipeline.py` - the rungs, and the one code path they share
- `src/gates.py`, `src/score.py`, `src/report.py` - validity, outcomes, the verdict
- `FINDINGS.md` - what the build turned up that the design did not predict

## Where this would fail

The `sametier` and `embedded` conditions are in the matrix because they are the
honest answer to "so is provenance the fix?". It is not. It is a decision
procedure over metadata, and it is exactly as good as that metadata.

An attacker who can write into a trusted tier, or who can get their text quoted
inside a document that is already trusted, gets the tier and the date working
for them instead of against them. Nothing in this repo detects that. The real
control there is write authorisation on the corpus, which is not a retrieval
problem at all.
