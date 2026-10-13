# understudy

Fine-tune a model on your own WhatsApp chat so it replies the way you do, on a
free Kaggle GPU, in about an hour, for nothing. Then measure what that actually
changed — because the answer is *how it sounds*, not *what it knows*, and most
people expect the opposite.

An understudy learns to perform as you. It has your delivery and none of your
memories. That is the whole project.

The interesting half is not the training run. It is turning a chat export —
bursts of four messages, `<Media omitted>`, invisible Unicode, two people
talking over each other — into pairs worth learning from.

## Setup

```
uv sync
cp .env.example .env        # then put your own name in ME=
```

There is no API key. Nothing in this project calls a paid service, and the
pipeline runs on a laptop CPU.

## Try it on the sample chat first

`data/sample_chat.txt` is a fictional export, written for this project. Nobody
real is in it. Run the whole thing on that before you point it at your own:

```
./demo.ps1          # or ./demo.sh
```

## The commands

```
PYTHONPATH=. uv run python -m src.main inspect --chat data/sample_chat.txt
PYTHONPATH=. uv run python -m src.main build   --me "Arjun Mehta"
```

`inspect` tells you who is in the file and which way round it writes dates.
`build` produces three files in `out/`:

| file | what it is |
|---|---|
| `train.jsonl` | the pairs the model learns from |
| `heldout.jsonl` | recent whole conversations it never sees |
| `stranger.jsonl` | the same, for the *other* person in the chat |
| `facts.json` | written by `facts check` — only the questions that passed |

That third file is the one that makes the result mean anything. See below.

Then the fact probe, which needs writing once per chat:

```
PYTHONPATH=. uv run python -m src.main facts scan   --me "Arjun Mehta"
PYTHONPATH=. uv run python -m src.main facts check  --me "Arjun Mehta"
```

`scan` lists things that were actually said and how often. You write questions
about the ones you can. `check` then refuses the bad ones — on the sample set
it throws out two of twenty-two, both because the answer only ever appeared in
a message that opened a conversation, so it never became a training pair at
all. A question whose answer was never in the training data cannot test recall.

It also refuses any question phrased too much like the chat itself. If the chat
said *"What is her name?"* and you probe with *"What is her name?"*, a hit shows
the model finishing a string it has seen, which is not the same as knowing
something.

## Then the GPU part

Upload the four files in `out/` to Kaggle as a dataset, open
`notebook/understudy_kaggle.ipynb`, set the accelerator to **T4 x2** and
internet to **on**, and run it. About twelve minutes of training.

The notebook writes `responses.json` and **renders no verdict**. It reports only
numbers a loaded model can produce. Every threshold lives on your laptop, so
changing your mind about what counts as a result does not cost another GPU
session.

```
PYTHONPATH=. uv run python -m src.main score --responses data/responses.json
PYTHONPATH=. uv run python -m src.main misses --responses data/responses.json
```

## The control is the argument

It is easy to fine-tune a model, watch it start saying "arre" and "bhai", and
conclude it learned *you*. It may only have learned what WhatsApp looks like —
short, lowercase, unpunctuated. Every chat looks like that.

So the same measurement runs twice: once over your held-out replies, once over
the *other person's*. If both improve by the same amount, the model learned a
medium, not a person. `score` prints both rows next to each other and says so.

The second control is the base model itself. Any fact the untuned model already
answers was never the fine-tune's to know, and is dropped from the count.

And style is the control for the fact result. If the voice did not move, the
training run did not take, and `score` refuses to report the fact numbers at
all — a model that learned nothing has forgotten nothing.

## Before you use your own chat

Read this bit.

**The other person is in your dataset too.** Their messages become the prompts.
Pick a chat you would be comfortable showing them, and do not use a group where
people would not expect it.

**Things do leave your machine.** The JSONL goes to Kaggle. `build` redacts
phone numbers, e-mail addresses, UPI handles and anything that looks like an
OTP before writing, and prints a count of what it caught — but it matches
Indian formats and English media markers, so check the output yourself. Keep
the Kaggle dataset private and delete it when you are done.

**`out/` is gitignored** and so is `data/my_chat.txt`. Keep it that way.

## What the first real run found

Ran end to end on a Kaggle T4 x2 on 2026-09-21. `data/responses.json` is that
run, committed, so `score` works before you have trained anything yourself.

**The voice moved, and it is not subtle.** Median reply length went from 240
characters (base) to 44 (tuned), against 40 for the real replies, and the
tuned model fits 40 of 40 held-out replies better.

**But most of that gain is not personal.** The control — the same measurement
on the *other* person's replies — captured 81% of it. The training mostly
learned this conversation's format and topics; the specifically-Arjun part is
about 19%. Reporting that honestly required fixing the metric, because
comparing the arms as percentages rather than nats had flattered the result
past a threshold it should not have cleared. See [FINDINGS.md](FINDINGS.md).

**Facts did not come along — and the failure is worse than a miss.** 1 of 20,
and that one appears six times in training, so it is a memorised string. The
rest are not blanks. Asked his rent, the tuned model said `12000` and invented
a service charge nobody mentioned; asked the Goa flight it produced an airline,
a flight number and a departure time, all wrong. The *base* model had answered
those correctly — "I'm an AI, I don't have a flat."

So the finding is sharper than the one this was built to show. Fine-tuning did
not fail to add knowledge. **It removed the refusal that was standing in for
the missing knowledge.**

Every module also checks itself, and none of it costs anything:

```
uv run python -m src.parse       # date formats, iOS vs Android, U+202F, BOM
uv run python -m src.sessionize  # gaps, bursts, alternation, held-out splits
uv run python -m src.clean       # what gets dropped, what gets redacted
uv run python -m src.grade       # the substring trap, negation, shotgunning
uv run python -m src.facts       # the gates
uv run python -m src.score       # the arithmetic
```

## What it will not do

It will not make the model know your life. That is the finding, not a
limitation to fix — and the guide spends its last section on why retrieval is
the tool for that job.

It will not work well on a small export. Under about 200 pairs the model
repeats whole lines back at you instead of generalising; `build` says so when
it counts them. The sample chat lands around 255, which is enough to watch the
voice move and not enough to be convincing. Your own export is probably bigger.
