# second-read

A local case workstation on MedGemma 1.5. Point it at a chest X-ray, a lab report PDF or a CT volume and it writes a case note — then re-asks every claim in that note with the image deleted, and again with another patient's image in its place.

Claims the model still makes without the image are marked **language prior, not evidence**.

Everything runs on one laptop GPU through Ollama. No API key, no HuggingFace gate, no account, no token budget.

---

## The catch this project is built around

A 4B model writes confident radiology prose whether or not it looked at the film. "The heart size is normal. The lungs are clear. There is no pneumothorax." — that paragraph is available to a language model that has never seen a chest X-ray, because it is what chest X-ray reports mostly say.

Nothing in the output distinguishes a reading from a recitation. So the product is not the note. The product is the note plus, for every claim in it, the answer to one question: *would it still have said this with the image deleted?*

That question has an answer you can compute, and it costs two extra forward passes per claim.

---

## The four conditions

Every condition asks the byte-identical question. Only the attached evidence changes.

| condition | the one switch | what it isolates |
|---|---|---|
| `sighted` | the case image attached | the product's own answer |
| `blind` | no image at all | does the claim survive deleting the evidence |
| `shuffled` | another patient's film, one a radiologist labelled the opposite way | does it survive *replacing* the evidence |
| `repeat` | seed 1 instead of 0 | the noise floor at temperature zero |

`shuffled` is not optional. Blind alone cannot tell "*this* image changed the answer" from "having *an* image changed the answer", and the middle case looks audited and is not.

The verdict falls out of the three: **PRIOR** when sighted and blind agree, **IMAGE_SHAPED** when the answer needs an image but not this one, **GROUNDED** when it differs from both.

Crossed with whether the claim is actually right, that gives the four cells the leaderboard reports:

| cell | meaning |
|---|---|
| `EARNED` | correct, and the image drove it |
| `LUCKY` | correct, but said with the image removed — right for the wrong reason |
| `PRIOR_ERROR` | wrong, and said with the image removed — what the flag should catch |
| `MISREAD` | wrong although the image drove the answer — **what the flag cannot catch, by construction** |

Printing `MISREAD` is the honest part. The audit measures whether the pixels drove a claim, not whether the claim is true.

---

## One case, end to end

```
$ PYTHONPATH=. uv run python -m src.main case CXR-0050

FINDINGS:
The heart is enlarged. The mediastinum is unremarkable. The lungs are clear.
There is no pleural effusion or pneumothorax. The bones are unremarkable.
IMPRESSION:
cardiomegaly.
```

| claim | in the note | with the film | with no film | with a stranger's film | verdict |
|---|:--:|:--:|:--:|:--:|---|
| cardiomegaly | yes | yes | **yes** | no | LANGUAGE PRIOR |
| consolidation | no | no | yes | no | image-shaped |
| pleural effusion | no | no | **no** | no | LANGUAGE PRIOR |
| pneumothorax | no | no | **no** | yes | LANGUAGE PRIOR |

Three of the four claims came back unchanged with the film deleted. The cardiomegaly call is correct — the reference report says cardiomegaly — and the model would have made it anyway.

---

## What the model never sees

`prompts.py` imports `CaseView`, never `Case`. A `CaseView` holds the images and the clinical indication; the reference report lives on `Case` alongside it. The reference has no type to travel in, so it cannot reach a prompt by accident — this is a structural guarantee, not a gate that runs and passes.

---

## The probe set

100 yes/no probes authored from IU-Xray (`ykumards/open-i`, 3,851 studies with free-text findings), over 7 findings, one probe per study, frontal view only.

Three rules make the blind floor an identity rather than an estimate:

1. **A "no" comes only from an explicit negation or an explicit statement of normal.** Unmentioned is *unknown*, never no. Building negatives from silence measures prevalence priors, and the label error is unfalsifiable.
2. **Balance per finding, not globally.** Each finding contributes exactly as many yes as no probes. A globally 50/50 set whose pneumothorax probes were 95% "no" would let the word predict the answer while the aggregate still read 50/50.
3. **The question text is byte-identical between a finding's yes and no probes.** The only difference is which image is attached.

Together: a blind model guessing any fixed answer per finding scores exactly 50%. Anything else is a leak, which makes it a gate rather than a result.

Labels come from a NegEx-style rule set cross-checked against the dataset's own manual MeSH index, then hand-checked in full. The checker may delete a probe or mark it uncertain — never flip a label to make balance work. Four rounds of that are written up in [FINDINGS.md](FINDINGS.md); every error it found was a rule gap producing many wrong labels, not a scattered mistake.

Three findings were dropped for having no balanced set to build: radiologists assert atelectasis, hyperinflation and degenerative change and almost never deny them.

---

## The synthetic lab reports

20 lab report PDFs, generated alongside their exact ground truth in one function so the two cannot drift, and extracted with the eight-field schema from MedGemma's own evaluation harness.

They exist to validate the audit machinery. The CXR probes are balanced so the blind floor is 50%; the lab probes have a floor of **zero**, because a value minted for one page exists nowhere else. A blind call that reproduces one is evidence the page reached a call that was supposed to have none — a check impossible to build on a chest film.

Getting that check right took two attempts, and the first failure is instructive. Scanning the blind answer for the *number* fails: asked to extract a report with no page attached, the model writes a six-thousand-character essay about how it would do the job, full of plausible example values, and a one-decimal blood urea nitrogen in a 7–20 range has only ~130 possible values. Two of 269 collided. So the gate reads the **structured** output instead and requires the value to arrive under its own analyte name. Guessing that some report contains 14.5 is free; naming it as this page's blood urea nitrogen is not.

---

## What the run found

100 probes × 4 conditions × 2 models, plus 20 lab pages, 35 shadow probes and 16 canary frames. Temperature 0, seeds 0 and 1, `num_ctx` 8192, images at 896 px.

| model | sighted | blind | shuffled | repeat agrees | flagged as prior |
|---|--:|--:|--:|--:|--:|
| `medgemma1.5:4b` | **75%** | 50% | **44%** | 100% | 57% |
| `gemma3:4b` | 58% | 51% | 54% | 100% | 77% |

| model | EARNED | LUCKY | PRIOR_ERROR | MISREAD | IMAGE_SHAPED |
|---|--:|--:|--:|--:|--:|
| `medgemma1.5:4b` | **24** | 41 | 16 | 5 | 14 |
| `gemma3:4b` | 6 | 43 | 34 | 5 | 12 |

**Five things this table says.**

**Blind landed on 50/100, exactly.** Not near the floor — on it. The probe design is doing what it claims, and every other number here can be read against a floor that is an identity rather than an estimate.

**MedGemma scores 44% shuffled, below chance.** Hand it another patient's film and it does *worse* than guessing, because it reads the film it is given correctly and the label belongs to someone else. The twin sits at 54%, which is chance — evidence it is not really reading either film. This is the clearest single sign that the specialist is looking and the generalist is reciting.

**The general model is more prior-driven, not less.** 77% of gemma3's claims survive deleting the image, against MedGemma's 57%. Medical tuning reduced the reciting by twenty points.

**EARNED is 24 against 6, where raw accuracy is only +17 points.** Sighted accuracy makes MedGemma look 1.3× better than the twin (75% vs 58%, exact McNemar p = 0.0396). Counting only claims that are correct *and* driven by the image, it is 4× better. A single accuracy number hides which of those two things you bought.

**LUCKY is the largest cell.** 41 of MedGemma's 100 claims are correct *and* would have been made with the film deleted. That is the finding the whole project exists to name, and it is why the criterion fails.

### The criterion: NOT MET

> The audit earns its place iff flagged claims are correct at least 15 points less often than cleared claims, with Fisher exact p < 0.05.

Flagged claims are 72% correct (95% CI 59–82). Cleared claims are 79% (65–89). The gap is **+7 points against a bar of 15**, Fisher exact **p = 0.49**. The flag does not separate wrong claims from right ones.

That is the failure mode this project pre-registered before running, and it is the honest result: on a corpus where "no acute abnormality" is usually true, **groundedness and correctness come apart**. A claim made without looking is not thereby a wrong claim. The audit tells you which claims are unearned; it does not tell you which are false, and a reader who conflates those two will trust it for something it cannot do.

### What the gates caught

Blind sits at 50% (CI 40–60). The canary reads 16/16 sighted and 0/16 blind. Every sighted row sent image bytes and every blind row sent none. Zero unstable rows in 100 at temperature 0. Balance survived attrition intact.

The lab floor is the sharp one: **blind recovers 0 of 269 minted values** under their own analyte name, while sighted recovers 168. Two loose numeric collisions remain in the blind prose and are reported beside the gate as collisions, not leaks — the story of why is [FINDINGS.md](FINDINGS.md) #13, and it is the most useful thing in this repo about designing a null probe.

### The number the shortcut would have published

Counting "the report never mentioned it" as "no" — the obvious way to build this probe set, and the wrong one — gives MedGemma **86%** on the shadow set against the honest **75%**. Eleven points of free inflation, available to anyone who does not check what a silent report actually means.

### The three preflight measurements

| question | answer |
|---|---|
| Does Ollama fuse two same-size images? (`ollama#17321`) | **No** for the Gemma 3 architecture — two 896² frames read back correctly, labelled or not |
| Does Q4 move MedGemma's bounding boxes? *(nobody had published this)* | **Yes.** Median IoU 0.677 between `q4_K_M` and `q8_0` across six films, range 0.29–0.87 — while each model against itself at two seeds is 1.0 |
| Does 8 images at 16K context fit 8 GB? | **Yes.** 2.7 GB resident, 0 GB spilled to CPU, 2,241 prompt tokens |

---

## What this project does not report

- **No IoU against truth.** MedGemma 1.5's largest measured gain over MedGemma 1 is bounding-box localization (3.1 → 38.0 on Chest ImaGenome), and Chest ImaGenome is PhysioNet-credentialed. There is no free gold box, so this repo reports box **stability** across seeds and box **swap-sensitivity** across patients, both of which compare the model with itself. A number invented to fill that hole would be the worst thing here.
- **No longitudinal benchmark.** MS-CXR-T is credentialed too. The comparison workflow ships with a prior-swap control instead: hand the model a stranger's film as the "prior" and count the interval changes it still reports.
- **No CT accuracy.** The slice-stack read runs and is labelled *demonstrated, not measured*.
- **The lab field accuracy is against PDFs we generated.** A real lab report is scanned, skewed, multi-column and half-legible. Treat the number as a ceiling, and do not compare it to the model card's 71.
- **IU-Xray is absent from MedGemma's *reported evaluation suite*.** That is not the same as absent from its pretraining corpus, and nobody outside Google can check the latter.

---

## Setup

Ollama 0.34 or newer, and about 7 GB of free VRAM.

```bash
ollama pull medgemma1.5:4b
ollama pull gemma3:4b

uv init second-read --python 3.12
cd second-read
uv add ollama pydantic pillow reportlab pymupdf nibabel numpy datasets rich python-dotenv
```

`gemma3:4b` is the same Gemma 3 decoder without the medical pretraining, at the same Q4_K_M quantisation. It is the control that isolates what the medical tuning bought.

No `.env` is required. There is no key.

---

## Run it

```bash
PYTHONPATH=. uv run python -m src.main models      # tags, digests, free VRAM, seconds per call
PYTHONPATH=. uv run python -m src.main build       # lab PDFs, probes, case folders
PYTHONPATH=. uv run python -m src.main gates       # every check that costs no inference

PYTHONPATH=. uv run python -m src.main note CXR-0050   # the product with the audit OFF
PYTHONPATH=. uv run python -m src.main case CXR-0050   # the same case, audited
PYTHONPATH=. uv run python -m src.main case ./my-folder

PYTHONPATH=. uv run python -m src.main locate  CXR-0050 --finding cardiomegaly
PYTHONPATH=. uv run python -m src.main compare PAIR-demo
PYTHONPATH=. uv run python -m src.main ct --slices 12

PYTHONPATH=. uv run python -m src.main run --profile full
PYTHONPATH=. uv run python -m src.main report
```

`case` is the one to run first: it prints the note, every claim lifted from it, all four probes per claim and the verdict. Every number the leaderboard reports is visible there for one case, which is the only way to believe the aggregate.

`case ./my-folder` accepts any directory. Modality comes from the extension and the prior from the filename (`prior`, `before`, `previous`, `baseline`, `old`); the reference is empty so the grading columns go blank and everything else works.

---

## Seven facts about Ollama that this code depends on

1. **`num_ctx` defaults to 4096 below 24 GB of VRAM**, and MedGemma's Modelfile sets only a stop token. The advertised 128K is a capability, not an allocation, and the overflow is silent. Every call here sets it.
2. **256 tokens per image, flat** — there is no pan-and-scan, so a 2,500 px film and an 896 px film arrive as the same 256 tokens. Images are resized to 896 before sending, and hashed after.
3. **Multiple images need one `[img]` placeholder each** in the prompt; Ollama substitutes them in order.
4. **A JSON schema in `format` works with images attached**, and it is what makes blind calls parseable.
5. **Constrained decoding suppresses the `<unused94>` thinking trace**, so anything needing reasoning runs unconstrained first and is structured in a second, text-only call.
6. **Box coordinates are `[y0, x0, y1, x1]`**, 0–1000 in the official notebook and 0–1 in the technical report. Parse defensively.
7. **Two 4B models will not co-reside in 8 GB.** The harness runs model-major and unloads between passes.

---

## Files

```
src/config.py     every path, tag and threshold, each with its reasoning
src/types.py      Evidence, CaseView, Case, Claim, Audit, Box - and the CaseView boundary
src/llm.py        the only module that talks to Ollama
src/images.py     resize, hash, PDF page to PNG, CT slice windowing, box overlays
src/claims.py     the finding vocabulary and the negation scanner both sides share
src/probes.py     authoring, MeSH cross-check, balancing, stamping
src/labgen.py     synthetic lab PDFs and their exact ground truth
src/extract.py    lab page to eight fields, scored exactly
src/audit.py      the four conditions, the single-switch check, the pure decision table
src/note.py       the note, its claims, and the audited markdown
src/boxes.py      localization, defensive parsing, stability and swap IoU
src/multi.py      longitudinal comparison, prior-swap control, CT slice stacks
src/gates.py      every validity check that costs no inference
src/harness.py    model-major, resumable
src/stats.py      Wilson, exact McNemar, Fisher exact, bootstrap - no scipy
src/report.py     the pre-registered criterion and the leaderboard
src/workstation.py  one case in, one audited record out
src/preflight.py  the three measurements nobody had published
```

---

## Where this would fail

The audit answers whether the pixels drove a claim, not whether the claim is true. It will hand you a well-grounded wrong reading with no warning at all — that is the `MISREAD` column, and it is printed for exactly that reason.

At temperature zero a blind call is effectively a constant per finding, so the *rate* of flagged claims is close to a property of the finding vocabulary rather than of any one case. The criterion is stated on whether flagged claims are less accurate than cleared ones, which can fail; a bar on the flag rate alone would be met by construction.

Seven findings is a narrow vocabulary. A note can only be audited for what the vocabulary can name, and anything the model says outside it passes through untagged.

---

## Not for clinical use

From MedGemma's model card, and it means what it says:

> MedGemma is not intended to be used without appropriate validation, adaptation, and/or making meaningful modification by developers for their specific use case. The outputs generated by MedGemma are not intended to directly inform clinical diagnosis, patient management decisions, treatment recommendations, or any other direct clinical practice applications. All outputs from MedGemma should be considered preliminary and require independent verification, clinical correlation, and further investigation through established research and development methodologies.

The model weights are under the [Health AI Developer Foundations terms](https://developers.google.com/health-ai-developer-foundations/terms), which define "Clinical Use" as any use in diagnosis or treatment of patients, including as part of a research study, and prohibit it. This project is a developer tutorial. It is a second read, never a first read, and in fact it is neither.
