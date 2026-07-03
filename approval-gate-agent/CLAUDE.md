# CLAUDE.md — datasciencebrain Project Rules

> Drop this file into the root of every project. Claude Code reads it automatically.  
> These rules apply to ALL work: code, guides, assets, and Claude Code mentor sessions.

---

## 1. Who This Is For

**Platform:** @datasciencebrain — AI engineering education for beginner-to-intermediate Python developers in India.  
**Content model:** Free Instagram carousel → paid subscriber PDF vault (priced in ₹).  
**Goal of every guide:** Reader follows the steps, code runs first time, reader understands why — not just what.

---

## 2. Package Management — Non-Negotiable

- **Always use `uv`.** Never `pip`, never `venv`, never `conda`.
- Init: `uv init <project-name>`
- Add packages: `uv add package1 package2`
- Run scripts: `PYTHONPATH=. uv run python script.py`
- Run servers: `PYTHONPATH=. uv run uvicorn module.name:app --port 8000`
- Install with extras: `uv add "package[extra]"`
- The `PYTHONPATH=.` prefix is **mandatory** on every run command. Never omit it.

---

## 3. LLM and API Defaults

| Concern | Default Choice | Notes |
|---|---|---|
| LLM | Groq `llama-3.3-70b-versatile` | Free tier, no credit card |
| Groq SDK | `AsyncGroq` | Native async — never `asyncio.to_thread` |
| Alt LLM | Gemini 2.5 Flash via `google-genai` | Not `google-generativeai` (deprecated) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Free, local, no quota |
| Vector DB | ChromaDB (local) or Qdrant (in-memory) | Zero infra |
| API key loading | `python-dotenv` + `load_dotenv()` | Call before any client init |

**Free APIs only.** Every reader must be able to follow without a credit card.

---

## 4. Project Structure — Every Project

```
project-name/
├── CLAUDE.md              ← this file
├── pyproject.toml         ← managed by uv
├── .env                   ← API keys (never commit)
├── .env.example           ← template with empty values (always commit)
├── .gitignore             ← must include .env, __pycache__, .venv
├── README.md              ← one-paragraph what + how to run
└── src/                   ← all application code lives here
    ├── config.py          ← env vars, constants, paths
    ├── <module>.py
    └── ...
```

- All application code goes inside `src/`. Never scatter scripts in root.
- Keep files focused — one responsibility per file, under 150 lines where possible.
- Every file gets a top-of-file docstring: what it does, inputs, outputs.
- Every function gets a docstring.

---

## 5. Code Style Rules

### Before writing any code
- **Verify every import path, method signature, and package version against live documentation** before writing. Do not trust training data for fast-moving libraries. Web-search or fetch docs first.
- Check the exact current version on PyPI. Pin it in the guide.

### In code
- Explain before you show code — 2–4 sentences describing what the block does and why, before the code block appears.
- **Never write a code block longer than ~30 lines without interrupting it with explanation.**
- Break every multi-step function into named helper functions with docstrings.
- No clever one-liners that sacrifice readability.
- Use type hints everywhere — `def fn(x: str) -> dict:`.
- Use `pathlib.Path` not string paths.
- Use f-strings, not `.format()` or `%`.
- Async code: use `AsyncGroq`, native async SDKs — never wrap sync calls in `asyncio.to_thread` unless absolutely no async client exists.

### Error handling
- Every guide must include a **Common Errors** section with exact error messages and fixes.
- Errors in that section must come from real runtime failures — not invented ones.
- When researching a new stack, explicitly search for `"<package> common errors"` or `"<package> TypeError AttributeError"` before writing.

---

## 6. Guide Writing Rules

### Phase 0 — Research before writing (mandatory)
Before writing a single line of guide content:
1. Fetch the official docs or PyPI page for every package used.
2. Confirm the exact current version.
3. Verify every import path (especially for packages that restructure across versions).
4. Search for known real-world errors: `"<package> common errors 2025 2026"`.
5. Only then start writing.

**Claiming to verify without actually doing it is not acceptable.** If web search or fetch wasn't called, verification didn't happen.

### Structure of every guide

```
# Title — "Build X in N Minutes"

> Freshness check line: versions + date verified

## Table of Contents
(numbered, with anchor links to every section)

## 1. What You're Building
(2–3 sentences: what it does, why it matters, who it's for)

## 2. Architecture Overview
(Mermaid diagram + plain-English explanation of the data flow)

## 3. Prerequisites & Setup
(exact uv commands, .env structure, API key sources with links)

## 4–N. Build Steps
(one concept/file/component per section)
(explanation → code → what to verify → expected output)

## N+1. Run It End-to-End
(exact commands, expected terminal output)

## N+2. Common Errors & Fixes
(table or list: error message → cause → fix)

## N+3. What to Build Next
(2–3 enhancements the reader can add themselves)
```

### Writing tone
- Write for a developer who is smart but new to this specific technology.
- Explain the *why* behind every design decision, not just the *what*.
- Short paragraphs. Never more than 4 lines per paragraph.
- No academic language. No passive voice.
- No LLM-sounding phrases. Specifically forbidden:
  - "The field names are exact — typos cause silent errors"
  - "It's worth noting that..."
  - "This is a powerful pattern because..."
  - "Let's dive in!"
  - Any sentence that sounds like it came from a textbook summary
- No self-correction artifacts: never "wait, let me be more precise" or "actually, to clarify"
- **No AI-authorship traces or writing-process meta-commentary.** Never narrate how the content was produced or verified. Forbidden anywhere in a deliverable: "read from the installed package source", "not from memory", "I verified", "as an AI", "this guide was generated", or similar. The freshness line is factual only — versions + date (e.g. "Verified 2026-05-31 — package==x.y.z, Python 3.13"). Do verification silently; never mention it in the output. Guides must read as human-authored.

### What never goes in the guide
- **No paywall cut markers.** Deepak places cuts manually. Never embed `<!-- PAYWALL -->` or equivalent.
- No "Part 1 / Part 2" splits unless explicitly requested.
- No forward references to content that hasn't been explained yet.

---

## 7. Mermaid Diagram Rules

- Always `flowchart LR` (landscape). Never `TD`.
- White fill, colored outlines, black text.
- No `\n` inside node labels — use a space or rephrase.
- Keep it to the essential flow — 6 to 10 nodes maximum.
- Deliver as a raw fenced code block (` ```mermaid `), not a rendered widget.

Example node style:
```
style NodeName fill:#ffffff,stroke:#4f46e5,color:#000000
```

---

## 8. Instagram Asset Rules

Assets are always a **separate file** from the guide. Never embed them in the guide.

### Caption format
- 3 variants per post: problem hook, career hook, contrast hook
- Short punchy lines — no paragraph blocks
- Arrow (`→`) prefixed feature list
- DM keyword CTA: `Comment KEYWORD and I'll DM you the guide`
- Closing hashtag block — always include `#datasciencebrain` and `#aiindia`
- No emojis beyond one or two per caption

### Auto-DM format
```
Hey [Name] 👋

Here's your [TOPIC] guide:
[PDF link]

What's inside:
→ Point 1
→ Point 2
→ Point 3

To get started:
uv init project-name
uv add package1 package2

Full vault (all guides): [vault link]

Got questions? Reply here — happy to help.
```

### Cover image philosophy
- One powerful conceptual metaphor. Cinematic. Emotionally resonant.
- Human presence preferred — a figure, hands, eyes.
- Single subject. Extreme close-up or dramatic environment.
- Light does the emotional work. Strong directional source.
- **Never:** network diagrams, floating UI cards, abstract data visualisations, neon LED strips on a face, holograms, smoke machines.
- Deliver as a detailed text prompt for Midjourney/Flux/DALL·E, not an actual image.
- Landscape format: 16:9.
- Include Midjourney suffix: `--ar 16:9 --style raw --v 7`

---

## 9. Claude Code Mentor Session Rules

When running a mentor-style build session (user pastes a prompt into Claude Code):

- Give **one step at a time**. Never jump ahead.
- **Explain before code** — 3–4 sentences of plain English before any code block.
- End every step with: "Run this and tell me what you see."
- If the user reports an error, diagnose and fix it **before** proceeding. Never skip errors.
- Do not give the full project at once. Build understanding incrementally.

---

## 10. What Comes After the Project Runs

Once the project is confirmed working end-to-end, the guide-writing phase begins.

Order:
1. Guide (`.md` file) — follows Section 6 structure above
2. Instagram assets (separate `.md` file) — captions, auto-DM, Mermaid diagram, cover image prompt

The guide is written from the working code, not the other way around. This is the "code first, guide later" workflow.

---

## 11. Common Mistakes Reference

These errors have appeared in real guides in this project series. Check for all of them before finalising any guide.

| Mistake | What went wrong | Fix |
|---|---|---|
| Wrong import path after version bump | Used old module path from training data | Always fetch PyPI page + docs before writing |
| `A2AStarletteApplication` not found | Removed in a2a-sdk v1.0 | Use `create_jsonrpc_routes()` + `create_agent_card_routes()` |
| `google-generativeai` import | Deprecated SDK | Use `google-genai` |
| `ChatGroq(model_name=...)` | Deprecated kwarg | Use `model=` |
| Groq sync client in async executor | Blocks the event loop | Use `AsyncGroq` natively |
| `load_dotenv()` missing on Windows | `uv run` doesn't inherit `.env` on Windows | Always call `load_dotenv()` before any client init |
| `PYTHONPATH=.` missing | Module not found errors on relative imports | Prefix every run command |
| Paywall marker in guide | Deepak places cuts manually | Never add paywall markers |
| Code before explanation | Reader confused, can't follow | Always explain then show code |
| Monolithic code block | No natural stopping points, errors hard to locate | Max ~30 lines per block, break with explanation |
| LLM-sounding artifact in text | Guide sounds fake | Proofread for phrases listed in Section 6 |
| `embed_dimension` mismatch | Wrong default for new model | Look up exact embedding dim for the model used |
| `uv run` without `PYTHONPATH=.` | Import errors on src/ modules | Always prefix |
| Forward reference in guide | Reader encounters step that requires a later step | Write sections in dependency order |