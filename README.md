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
| [visual-search](visual-search/) | Visual product search using CLIP |
| [langsmith-tutorial](langsmith-tutorial/) | LangSmith observability tutorial — tracing, datasets, and LLM-as-judge evaluations |
| [rag-eval](rag-eval/) | RAG evaluation pipeline using RAGAS v0.4 — faithfulness, answer relevancy, context precision, and recall scored via Gemini 2.5 Flash |
| [memory-agent](memory-agent/) | Persistent memory chat agent — stores and retrieves user facts across sessions using mem0, Groq, and local Qdrant |
| [parallel-news-agent](parallel-news-agent/) | LangGraph map-reduce pipeline that researches multiple news topics in parallel and assembles them into an executive briefing using Groq and Tavily |

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