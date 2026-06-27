# Reflexion Self-Correcting Agent

A cold-email writer that grades and rewrites its own work. A **generator** drafts the
email, a cheap **critic** scores it against a fixed rubric, and a cyclical LangGraph loop
feeds the critique back until the score passes a threshold or hits a hard revision cap.
A one-shot **adjudicator** on a stronger model gives the final verdict, and the run plots
a **diminishing-returns curve** so you can see the revision where quality stops improving.

```
START → generator → critic → pass? ──no──► loop back to generator (until MAX_REVISIONS)
                       │
                      yes / cap reached
                       ▼
                  adjudicator → END
```

## Three-tier model split

| Role | Model | Runs |
|---|---|---|
| Generator (writes the email) | `gemini-3.5-flash` | every revision |
| Critic (scores + feedback) | `gemini-3.1-flash-lite` | every loop — cheap on purpose |
| Adjudicator (final verdict) | `gemini-3.1-pro` | once, at the end |

`gemini-3.5-pro` is enterprise-preview only for now; swap it in for the adjudicator in
`src/config.py` once it reaches general availability.

## Setup

```bash
cp .env.example .env        # then paste your key from https://aistudio.google.com/apikey
uv sync
```

## Run

```bash
PYTHONPATH=. uv run python -m src.main
PYTHONPATH=. uv run python -m src.main "your own outreach brief here"
```

You'll see each handoff stream by (draft #1 → score → revise → draft #2 …). When it
finishes:

- `output/email.md` — the final email plus the adjudicator's verdict
- `output/curve.png` — score vs. revision, with the pass threshold and the knee marked

## Layout

```
src/
├── config.py   # models, threshold, revision cap, LLM factory
├── state.py    # Critique schema + ReflexionState
├── prompts.py  # shared rubric for generator and critic
├── nodes.py    # generator / critic / adjudicator + routing
├── graph.py    # the cyclical StateGraph
├── plot.py     # diminishing-returns curve
└── main.py     # CLI entry point
```

## Tuning

- `PASS_THRESHOLD` and `MAX_REVISIONS` in `src/config.py` control how strict the loop is
  and how many rounds it may run.
- A harder/vaguer brief forces more revisions and a more visible knee in the curve.
