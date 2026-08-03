# Daily AI Mini Projects

A collection of AI/ML mini projects exploring various AI capabilities.

## Projects

| Project | Description |
|---------|-------------|
| [agent-memory-router](agent-memory-router/) | LLM-powered router that classifies queries to RAG, graph, or tabular backends |
| [ai-vani](ai-vani/) | AI voice assistant |
| [annual-report-ai](annual-report-ai/) | Vision AI for reading and analyzing annual reports |
| [chat-with-csv](chat-with-csv/) | Chat with CSV data |
| [desibot](desibot/) | Desi-focused chatbot |
| [github-mcp](github-mcp/) | GitHub MCP server implementation |
| [graphrag-project](graphrag-project/) | Graph-based RAG with knowledge graphs |
| [hybrid-rag](hybrid-rag/) | Hybrid RAG combining vector search (ChromaDB) and graph traversal (NetworkX) |
| [marketmind](marketmind/) | Multi-agent market analysis system |
| [multimodal-agent](multimodal-agent/) | Agent with image understanding capabilities |
| [lightrag-project](lightrag-project/) | LightRAG-based knowledge graph retrieval with dual-level retrieval |
| [pageindex-rag](pageindex-rag/) | Reasoning-based RAG over PDFs using PageIndex tree navigation (no vectors) |
| [sarvam-chatbot](sarvam-chatbot/) | Sarvam AI chatbot |
| [self-healing-rag](self-healing-rag/) | RAG system with self-correction capabilities |
| [telegram-bot](telegram-bot/) | Telegram bot with memory engine |
| [video-rag](video-rag/) | RAG system for querying video content via frame extraction and embeddings |
| [vision-rag](vision-rag/) | Vision-based RAG over PDFs using image embeddings for visual document understanding |
| [voice-support-bot](voice-support-bot/) | Voice support bot with token server |
| [voice-ai-agent](voice-ai-agent/) | Real-time voice assistant on LiveKit Agents with swappable STT / LLM / TTS presets |
| [visual-search](visual-search/) | Visual product search using CLIP |
| [langsmith-tutorial](langsmith-tutorial/) | LangSmith observability tutorial — tracing, datasets, and LLM-as-judge evaluations |
| [rag-eval](rag-eval/) | RAG evaluation pipeline using RAGAS v0.4 — faithfulness, answer relevancy, context precision, and recall scored via Gemini 2.5 Flash |
| [memory-agent](memory-agent/) | Persistent memory chat agent — stores and retrieves user facts across sessions using mem0, Groq, and local Qdrant |
| [parallel-news-agent](parallel-news-agent/) | LangGraph map-reduce pipeline that researches multiple news topics in parallel and assembles them into an executive briefing using Groq and Tavily |
| [a2a-agents](a2a-agents/) | Multi-agent system on the A2A protocol — Research and Writer agents discover each other via Agent Cards and chain over JSON-RPC, powered by Groq |
| [computer-use-agent](computer-use-agent/) | Desktop control agent — Gemini Flash sees the screen and decides actions in a LangGraph See→Think→Act loop, executed with PyAutoGUI |
| [self-healing-code-agent](self-healing-code-agent/) | Code agent that writes Python, runs it in a subprocess, reads its own tracebacks, and fixes its own bugs until self-written tests pass — built on Pydantic AI and Groq |
| [smolagents-code-agent](smolagents-code-agent/) | smolagents CodeAgent powered by Groq that solves any problem by writing and running Python — supports web search, custom tools, and pandas/numpy in generated code |
| [live-vision-agent](live-vision-agent/) | Real-time agent that streams webcam + microphone to Gemini Live API over a single WebSocket session and responds in natural speech with live terminal transcript |
| [realtime-voice-translator](realtime-voice-translator/) | Speak any language into your mic — Gemini Live Translate API translates and speaks back in your chosen target language, live, with a Streamlit browser UI |
| [interview-coach](interview-coach/) | Real-time AI interview coach powered by Gemini Live — watches you via webcam and coaches tone, pacing, filler words, body language, and content after each answer |
| [worldcup-analyst](worldcup-analyst/) | LangGraph parallel multi-agent World Cup 2026 analyst — three specialist agents fan out simultaneously with Send, each gather their own data slice, then a synthesizer assembles a match briefing using Groq and Tavily |
| [supervisor-content-team](supervisor-content-team/) | LangGraph supervisor multi-agent content team — a routing supervisor coordinates researcher, writer, editor, and SEO specialists via shared state and structured output, powered by Gemini and Tavily |
| [colpali-vision-rag](colpali-vision-rag/) | Multimodal RAG that answers questions about charts, tables, and scanned PDFs by looking at the pages — no OCR. ColPali (ColQwen2) embeds each page into patch-level multivectors, Qdrant ranks them with MaxSim late interaction, and Gemini 3.5 Flash reads the retrieved page images to answer, wired with LangGraph |
| [reflexion-self-correcting](reflexion-self-correcting/) | Reflexion self-correcting agent — a generator writes a cold email, a cheap critic scores it against a rubric, and a cyclical LangGraph loop feeds the critique back until quality passes or hits a revision cap, then a pro-tier adjudicator gives the final verdict and the run plots the diminishing-returns curve. Three-tier Gemini split (3.5-flash / 3.1-flash-lite / 3.1-pro) |
| [agent-eval-arena](agent-eval-arena/) | Head-to-head eval harness — the same tool-calling agent runs across three Groq models on a fixed test set, Langfuse traces every call, exact-match and an LLM judge score each answer, and a leaderboard ranks accuracy, cost, and latency |
| [approval-gate-agent](approval-gate-agent/) | Human-in-the-loop approval gate — a LangGraph agent pauses on risky actions with interrupt(), persists its state to a durable SQLite checkpoint, and resumes exactly where it stopped once you approve, even across separate processes, powered by Groq |
| [rag_anything_app](rag_anything_app/) | Multimodal document chat — RAG-Anything ingests PDFs with text, tables, and images, embeds them locally with bge-m3, and answers questions over a free Gemini-compatible endpoint through a Gradio UI |
| [offline-slm-agent](offline-slm-agent/) | Fully offline tool-calling agent — a hand-built ReAct loop drives a local Ollama model (qwen3:4b) through calculator and sandboxed file tools with zero network calls, proven offline via netstat |
| [plan-and-execute-agent](plan-and-execute-agent/) | Plan-and-execute agent — a LangGraph planner drafts a multi-step plan, an executor works each step, and a replanner adapts as it goes, benchmarked against a plain ReAct agent over keyless Wikipedia and calculator tools on Groq |
| [lora-finetune-kaggle](lora-finetune-kaggle/) | QLoRA fine-tune on free Kaggle GPUs — turns Qwen3-4B into "DesiTutor", a Hinglish coding tutor, with 4-bit quantization, PEFT LoRA adapters, and a before/after evaluation |
| [faq-autopilot](faq-autopilot/) | Long-horizon agent that watches a docs folder, detects drift when a file changes, and updates a customer FAQ on its own — every answer cited back to its source, every action logged, with durable SQLite state, on free Groq gpt-oss-120b |
| [deep-research-team](deep-research-team/) | Deep-research multi-agent team on LlamaIndex Workflows — a planner fans sub-questions out to four researchers running concurrently, a reflector spots gaps and sends the team back out, and a writer produces a cited report, on free Gemini with keyless DuckDuckGo search |
| [personal-ai-assistant](personal-ai-assistant/) | A real tool-calling agent that lives in Telegram — routes every message through an LLM to web search, notes, reminders, and article/YouTube/PDF summaries, messages you first with a durable morning briefing, and remembers across restarts via a SQLite-backed LangGraph checkpointer, on free Groq gpt-oss-120b |
| [mcp-host](mcp-host/) | An MCP host that keeps five Model Context Protocol servers connected at once and chains tool calls across all of them from a single sentence — built on the mcp 2.0 SDK's ClientSessionGroup, with three official uvx servers (time, fetch, git) plus a sandboxed files server and a sqlite notes server you write yourself, a per-server tool allowlist to stay inside the free-tier context budget, and a Streamlit UI over a background event loop, on free Groq gpt-oss-120b |

## Getting Started

Each project has its own setup instructions. Generally:

```bash
# Python projects
cd <project-name>
pip install -e .

# Node.js projects
cd <project>/frontend
npm install
```

## Requirements

- Python 3.10+
- Node.js 18+