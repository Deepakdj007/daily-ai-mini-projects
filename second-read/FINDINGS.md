# Build findings

Things the build turned up that the design did not predict. Written down as they happened, so the README and the guide's Common Errors section quote real runtime behaviour rather than reconstructed memory.

Grouped: how the model and Ollama actually behave, then getting parseable answers out of them, then building the probe set, then what the run found.

---

## Model and runtime mechanics

## 1. MedGemma 1.5 on Ollama emits a bare `<unused94>` before the answer, with no closing token

The official notebooks say a thinking trace sits between `<unused94>` and `<unused95>` and the answer follows the second token. On `medgemma1.5:4b` through Ollama 0.34, a trivial prompt returned exactly `'<unused94> ready.'` — four tokens, `done_reason: stop`, no `<unused95>` anywhere. Stripping code that treats the open token as "a trace has started" and keeps only what precedes it returns an empty string for every call, and nothing errors. The fix is to remove a closed trace when one exists and otherwise delete the lone token and keep the rest.

## 2. Ollama's 4,096-token default context is real on this card and the Modelfile does not lift it

`ollama show medgemma1.5:4b` lists `stop <end_of_turn>` as the only parameter. The documented default context below 24 GB of VRAM is 4k. The advertised 128K is what the model can do, not what Ollama allocates. Every call in `llm.py` passes `num_ctx` explicitly.

## 3. Cold load with a concurrent pull is 70 seconds

The first call to `medgemma1.5:4b` took 70.6 s end to end while `gemma3:4b` was still downloading to the same disk. Warm calls on the same 17-token prompt are ~0.1 s. Any timing the `models` preflight reports has to be a warm call, and the harness has to expect a one-off load per model swap.

## 4. Ollama does not merge same-size images for the Gemma 3 architecture

`ollama#17321` reports adjacent same-size images being fused into video frames for Qwen-VL. Two 896×896 frames reading FRAME ALPHA and FRAME BRAVO, sent to `medgemma1.5:4b` positionally with no labels, came back as "two separate images", each transcribed correctly — and the same with `[img]` placeholders, and with a one-pixel size difference. The labels stay in the prompts because they cost nothing and address the separate image-bleed report, but the merge bug is not this architecture's problem on Ollama 0.34.

## 5. Eight images at 16K context sit entirely on the 8 GB card

`ollama ps` after each call: 2.7–2.8 GB on GPU, 0 GB on CPU, at `num_ctx` 8,192 and 16,384, with one image and with eight (2,241 prompt tokens). The eight-image call took 13 s including a cold load. Twelve CT slices are within reach; Google's 85 would not be.

## 6. The model quotes its own image delimiters back as text

Asked to compare a prior and a current film, the prose came back with `**Prior Study:** <start_of_image> <end_of_image>` in it — the model narrates the placeholder tokens when it refers to "the first image". Harmless to the answer, ugly in a rendered note; the token strip in `llm.py` removes them along with `<unusedNN>`.

---

## Getting a parseable answer out

## 7. Free-text answers with a "Final Answer:" anchor lose nine blind calls in twenty-one

Measured on 21 probes, sighted and blind. With free generation at 512 new tokens the blind calls parsed 12 of 21: asked whether cardiomegaly is present with no image attached, the model writes a textbook paragraph about what cardiomegaly looks like, hits the cap and never reaches its answer. Raising the cap to 1024 got 18 of 21 and pushed the mean blind call to 8.2 s. A JSON schema with `reasoning` first and a `final_answer` enum second parsed 21 of 21 both ways, cut the blind call to 1.1 s, and left sighted accuracy unchanged at 14 of 21. The reasoning field has to come first — a bare enum makes the model answer from nothing.

Ollama's `format` parameter takes a full JSON Schema, and it works with images attached.

## 8. The image canary works, and a blind call answers nothing

A grey frame with `CASE ID KQ-7741` printed on it: sighted read `KQ-7741`, blind produced no Final Answer line at all. That pair is the gate that proves image bytes reach the model. Blind will sometimes invent an id instead of declining, so the gate scores the string, not the presence of an answer.

---

## Building the probe set

## 9. The open-i parquet shards can be read by column over HTTP

`ykumards/open-i` is five ~420 MB parquet shards, 771 rows each in 100-row groups. `HfFileSystem` plus `pyarrow.parquet.ParquetFile` reads the eight text columns of one row group in ~4 s and one row group's frontal images in ~8 s, so the probe author scans every report without downloading 2.1 GB and only pulls the images of the studies it keeps.

## 10. The claim rules needed four rebuilds, and every fix was systematic

Hand-checking the authored probes turned up errors in batches of 9, then 6, then 3, then 7 — and almost all of them were one rule gap producing many wrong labels, not scattered mistakes:

- `clear of` was missing from the negation cue list. IU-Xray's commonest way of denying airspace disease is "the lungs are clear of focal airspace disease", and seven of those were labelled as assertions.
- "Is a rib fracture present" is not answered by a healed clavicle. The fracture pattern matched any bone; four probes asserted a rib fracture on the strength of a humerus, a clavicle and a T6 vertebra.
- A negation carrying a **size** qualifier denies only that grade: "no large pleural effusion" is silent about a small one. But "no focal consolidation" and "no acute displaced rib fracture" are how radiologists write a plain negative — treating those as hedges too emptied four findings of their negatives on the first attempt.
- Alternation in an assertion commits to nothing: "small area of atelectasis **or** infiltrate". The same "or" inside a negation denies both, so the rule only fires where nothing is negated.
- Vascular congestion is not pulmonary edema. Three probes were labelled yes on "mild pulmonary vascular congestion", while every negative said "edema" — including congestion on the positive side only would have tilted the finding.

The dataset's own manual MeSH index is the second opinion that caught most of these: a report that says "no nodules or masses" on a study indexed under `Granuloma/lung` is contradicting itself, and neither side of that is a probe.

## 11. Seven findings survive per-finding balancing; three do not

Of the ten findings in the vocabulary, `atelectasis` (161 explicit assertions, 2 explicit negations), `hyperinflation` (189/1) and `degenerative` (476/1) are dropped. Radiologists assert them and almost never deny them, so there is no balanced set to build and no honest floor. Pneumothorax survives with 15 assertions against 2,440 negations — the opposite imbalance, and the reason per-finding balancing matters: a globally 50/50 set whose pneumothorax probes were 95% "no" would let the word predict the answer while the aggregate still read 50/50.

## 12. A span-based out-of-range rule mints two-digit lab values

The first version of the lab generator pushed a "low" value below the range by a fraction of the range's width. For ferritin (30–400) that produced `5.1` — a value short enough to be guessable and to collide with other numbers on the page. Out-of-range values now scale with the boundary they cross.

## 13. A minted lab value is not unguessable just because it is precise

The synthetic lab reports exist to give the audit a floor of exactly zero: a value minted for one page and appearing nowhere else cannot be produced by a call that never saw the page. On the first full run the gate failed — **2 of 269 values were "recovered" blind**, on pages where `image_bytes_sent` was 0 and the canary confirmed no image reached the model.

Neither was a leak. Asked to extract a lab report with no page attached, MedGemma 1.5 writes a 6,361-character essay about how it *would* do the job, thick with plausible example numbers. Two minted Blood Urea Nitrogen values, `14.5` and `10.0`, appear in that essay by coincidence. A one-decimal BUN inside a 7–20 reference range has roughly 130 possible values, not the "thousands" the design assumed, and the essay contains enough numbers that a collision across 269 values is expected rather than alarming.

Raising the precision does not fix it — pushing to five or six significant digits buys an answer space large enough, at the cost of values no real lab would print. The fix is to change what the gate reads: a value only counts as recovered when it arrives **attached to its own analyte name** in the structured output. Guessing that some lab report somewhere contains 14.5 is free; naming it as the blood urea nitrogen of a page you were never shown is not. The loose prose count is kept as a diagnostic beside it, labelled as collisions.

Once corrected, the instrument separates cleanly: blind recovers **0 of 269** values under their own name, sighted recovers **168**.

The general lesson is the one this whole project is about: a null probe is only as good as the thing it actually measures, and "the value appears in the text" and "the model read the value off the page" are not the same claim.

---

## What the run found

## 14. Q4 moves MedGemma's bounding boxes, and nobody had measured it

The same localization prompt on six chest films at `medgemma1.5:4b` (Q4_K_M, 3.3 GB) and `medgemma1.5:4b-it-q8_0` (5.0 GB): median IoU between the two quantisations is **0.677**, ranging 0.286 to 0.873. The boxes agree on roughly where the heart is and disagree on its extent — Q8 consistently starts the box lower. Both are self-consistent: the same model at two seeds gives IoU 1.0. Coordinate regression through language-model logits is more quantisation-fragile than captioning, which is what this measures. Q8 also costs about 30% more wall clock and still fits 8 GB.

## 15. The bounding box moves between patients, but only halfway

On a real IU-Xray film, `locate --finding cardiomegaly` returns the same box at both seeds — self-IoU **1.0**, so greedy decoding is genuinely deterministic here. Handed a film a radiologist labelled negative for cardiomegaly, the box moves, but only to IoU **0.56** against the original.

So the box is neither a fixed prior over where hearts live (which would give ~1.0) nor a fresh reading of each film (which would give much less). It is anchored on anatomy and adjusted somewhat by the image. Reading the overlay confirms it: the box is roughly right and distinctly loose, covering mediastinum and inner lung fields alike.

There is no gold box to score this against — Chest ImaGenome is credentialed — so stability and swap-sensitivity are the only two honest numbers available, and they say more than a single IoU would.

## 16. The structured comparison invents a resolution

On Google's own longitudinal demo pair the free-text comparison was sensible, and the schema-constrained second pass still emitted `pneumothorax: resolved` for two films that never showed one. Absent-then-absent is "stable" or "not assessable"; "resolved" is a change that never happened. This is precisely the class of claim the prior-swap control exists to count.

## 17. Every interval change survived swapping the prior for a stranger

`compare PAIR-demo` on Google's own two public demo films reports heart size stable, lung opacities worsened, pleural effusion resolved, pneumothorax resolved. Run again with an unrelated patient's chest X-ray substituted for the prior, the model returns **the identical four verdicts**. Three of them are non-trivial changes, and all three are confabulated: nothing in the "prior" it was shown supports them.

This is the honest residue of a capability that cannot be benchmarked for free: MS-CXR-T is PhysioNet-credentialed, so there is no scored longitudinal number in this repo. The prior-swap control costs one extra call and catches the failure anyway.

## 18. The one prediction that was wrong: shuffled fell below chance, not between

Four predictions were written into `ladder.py` before the run. Three held: sighted above 70% for the subject (75%) and nearer 55% for the twin (58%); blind at the 50% floor for both (50% and 51%); repeat disagreeing on under 5% of rows (0 of 100).

The fourth was wrong. `shuffled` was predicted to land *between* blind and sighted, on the reasoning that a stranger's film is worse evidence than the right one but better than none. MedGemma came in at **44%** — below the 50% blind floor, and below chance.

The explanation is the interesting part, and it is evidence the model is working rather than failing. On a balanced set, answering at 50% is what you get by ignoring the image. Scoring *below* 50% means the answers are anti-correlated with the labels, which happens when the model reads the film it was given accurately and the label belongs to a different patient. The donor rule guarantees that: every shuffled film is one a radiologist labelled the opposite way. A model that genuinely reads gets systematically marked wrong.

The twin sits at 54%, indistinguishable from chance — which says it is not really reading either film. So the sub-chance number is the sharpest single signal in the run that the specialist looks and the generalist recites, and it arrived from a prediction that was wrong.
