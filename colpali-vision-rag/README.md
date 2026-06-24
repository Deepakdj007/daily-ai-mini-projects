# ColPali Vision RAG — read charts & scanned PDFs without OCR

A multimodal RAG that answers questions about **charts, tables, and scanned pages**
by *looking* at them — no OCR, no text extraction. ColPali (ColQwen2) embeds each
PDF page as an image into patch-level multivectors, Qdrant ranks pages with MaxSim
late interaction, and Gemini 3.5 Flash reads the retrieved page images to answer.
A small LangGraph flow wires retrieve → answer.

```
PDF pages → ColQwen2 visual multivectors → Qdrant (MAX_SIM) → retrieve page images → Gemini 3.5 Flash answers
```

## Setup

```bash
uv sync                       # installs deps incl. CUDA torch (cu128)
cp .env.example .env          # then paste your free Gemini key
```

Needs: an NVIDIA GPU (CPU works but is slow) and **Poppler** for PDF rendering.
Set the Poppler path in `src/config.py` (`POPPLER_PATH`).
Get a free Gemini key (no card) at https://aistudio.google.com/apikey.

## Run

```bash
# 1. make a sample image-only PDF (a bar chart + a data table)
PYTHONPATH=. uv run python scripts/make_sample_pdf.py

# 2. embed every page into Qdrant
PYTHONPATH=. uv run python src/ingest.py

# 3. ask questions answered straight off the page images
PYTHONPATH=. uv run python src/main.py "What was revenue in Q3?"
PYTHONPATH=. uv run python src/main.py "Which region had the highest revenue?"
```

Point it at your own files by dropping PDFs into `pdfs/` and re-running step 2.
