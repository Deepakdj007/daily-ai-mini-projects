# Build findings

Things this build turned up that the design did not predict, written down as
they happened so the README and the guide's Common Errors quote real behaviour
rather than reconstructed memory.

First full Kaggle run: 2026-09-21, Tesla T4 x2, 212 training pairs, 60 steps.

---

## 1. The tuned model does not miss facts. It invents them, fluently.

This project was built to show that fine-tuning moves style and does not
install knowledge. The expected result was a table of misses. What actually
happens is worse, and is the finding the guide should be built on.

The **base** model refuses correctly:

> *Which two-wheeler do you ride around town?*
> — "I don't ride any two-wheeler since I'm an AI and don't have physical form"

> *You keep a pet at home. What did you decide to call her?*
> — "I don't have a pet at home, so I don't have a name for one! 😊"

The **tuned** model answers confidently, in voice, and is wrong every time:

| asked | truth | tuned model said |
|---|---|---|
| two-wheeler | Access 125 | `pulsar 150` / `bhai wo toh 150 ka hi hai` |
| monthly rent | 18500 | `12000` / `aur 10000 ka service charge` |
| who owns the flat | Mr. Raghavan | `mummy ka` |
| Goa flight | 6E 2134 | `indigo 2212` / `10:30 am flight` |
| wifi password | jamun4590 | `waise bata de` / `waise bata de` |

It did not merely fail to learn the rent. It invented a rent *and a service
charge that was never mentioned by anyone*. It invented an airline, a flight
number and a departure time.

So the honest statement is not "fine-tuning does not add knowledge". It is:
**fine-tuning removed the refusal that was standing in for the missing
knowledge.** The base model's "I'm an AI, I don't have a flat" was doing real
safety work, and 60 steps of SFT on someone's chat deleted it.

1 of 20 facts came back (`Kabini`, which appears 6 times in training) — and a
string seen six times being reproduced is memorisation, not recall.

## 2. Comparing the control as a percentage flattered the result by 4x

The first verdict printed *"the gain is about you and not about WhatsApp in
general"*. That was wrong, and it was wrong because of the unit.

```
              base    tuned   fell by   as %
yours         6.026   2.662   3.363    +55.8%
the control   7.512   4.800   2.712    +36.1%
```

Read as percentages, the arms look well separated: 55.8 against 36.1. The
threshold was "yours must beat the control by 1.5x", and 55.8 clears
36.1 x 1.5 = 54.2 — by three points.

But a percentage is relative to each arm's own baseline, and the baselines are
not the same. The control **starts worse** (7.512 vs 6.026, because Nikhil
writes longer, more varied English while Arjun's replies are short and
formulaic), so an identical drop reads as a smaller percentage for him. In
nats, which is the unit NLL is already in, the control captured
**2.712 / 3.363 = 81%** of the gain.

The corrected reading: the training mostly learned *this conversation* — its
format, its topics, its names — and the part that is specifically Arjun is
about 19% of the effect. Real, but the minority.

The fix is in `score.py`: compare the `fell by` column, never the percentages,
and report the control's share rather than passing a threshold. The old
threshold is the kind that looks like a pass and is not one.

Worth saying that this is a *hard* control — same chat, same topics, same
format, only the person differs. A stranger from an unrelated chat would have
moved much less, and would have made the result look better for no honest
reason.

## 3. Style transfer itself is not in doubt

Separately from the specificity question, the format shift is unambiguous:

| | median reply length |
|---|---|
| Arjun's real replies | 40 chars |
| base model | 240 chars |
| tuned model | 44 chars |

And the tuned model fits **40 of 40** held-out replies better than the base
does. The base answers *"Sure! Let's break it down. You said: **'How was
it?'**..."*; the tuned model answers `haan` / `but wo sab alag lag rahe hain`.

## 4. Two of twenty-two fact questions were unusable, and the gates caught both

`f18` (2400, money transferred) and `f21` (1800, scooter battery) were both
written from the chat in good faith. Both were rejected as *"never said in the
training split"* — because in each case Arjun **opened** the conversation with
that line, and a session-opening message has no preceding turn, so it never
becomes a training pair at all.

A question whose answer was never in the training data cannot test recall, and
counting its miss would have padded the denominator. The notebook is fed
`out/facts.json`, the gated subset, not `data/facts.json`.

## 5. Runtime facts from Kaggle worth pinning

- Unsloth 2026.9.7 / Transformers 5.5.0 / Torch 2.10.0+cu128 / Triton 3.6.0.
- **`Bfloat16 = FALSE`** on the T4. Anything bf16-only is not an option here.
- Unsloth **shards the 4B across both T4s by itself** (weights 2.089 GiB on
  cuda:0, 1.168 GiB plus the output head on cuda:1). The banner says
  `Data Parallel GPUs = 1` even so — that is model parallelism, not one idle
  GPU.
- `train_on_responses_only(trainer)` auto-detected
  `'<|im_start|>user\n'` / `'<|im_start|>assistant\n'` correctly and masked to
  9 of 27 tokens (33%). The silent all-`-100` failure did not fire on this
  template; the assertion stays because it costs nothing and a template change
  would bring it back.
- 60 steps took **5 min 29 s**, not the twelve estimated. 66,060,288 trainable
  parameters of 4,088,528,384 (1.62%).
- `Both max_new_tokens and max_length seem to have been set` prints **once per
  `generate()` call** — about sixty lines of it. Harmless.

## 6. The GGUF export is not named what you would guess

`save_pretrained_gguf("understudy_gguf", ...)` writes to
**`understudy_gguf_gguf/`** — the suffix is doubled — and the file inside is
named after the base model, `qwen3-4b-instruct-2507.Q4_K_M.gguf`. Nothing in
the filename indicates it contains a fine-tune.

The first draft of this notebook shipped a hand-written `Modelfile` with
`FROM ./understudy.Q4_K_M.gguf`, which matches no file that is ever produced.
Unsloth writes a correct `Modelfile` beside the GGUF; use that one.

Also worth knowing: conversion frees the 16-bit intermediates mid-run because
only 4.2 GB of disk was free against ~4.6 GB needed. llama.cpp now installs
prebuilt (b11030) rather than compiling.
