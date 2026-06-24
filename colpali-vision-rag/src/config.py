"""Central configuration: env vars, model names, paths, and Qdrant settings.

Everything tunable lives here so the rest of the code reads cleanly.
Loads the .env file once on import so GEMINI_API_KEY is available everywhere.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Project paths -----------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
PDFS_DIR = ROOT_DIR / "pdfs"
PAGE_IMAGES_DIR = ROOT_DIR / "page_images"
QDRANT_PATH = ROOT_DIR / "qdrant_data"

# --- Poppler (pdf2image backend on Windows) ----------------------------------
# pdf2image shells out to Poppler. On Windows it is not on PATH by default,
# so we point at the installed binaries. Set to None on Linux/macOS.
POPPLER_PATH = r"C:\poppler\poppler-26.02.0\Library\bin"

# --- Visual retrieval model (ColPali family, runs locally) -------------------
# colqwen2-v1.0 is the ~2B model that fits comfortably in 8 GB VRAM.
# For higher chart/table accuracy on a bigger GPU, swap to "vidore/colqwen2.5-v0.2".
COLPALI_MODEL = "vidore/colqwen2-v1.0"

# Pages are rendered at this DPI before embedding. 150 is a good speed/quality mix.
RENDER_DPI = 150

# --- Qdrant multivector store ------------------------------------------------
COLLECTION_NAME = "pdf_pages"
# ColPali/ColQwen emit one 128-d vector per image patch (a multivector per page).
VECTOR_DIM = 128
TOP_K = 3

# --- Vision answer model -----------------------------------------------------
GEMINI_MODEL = "gemini-3.5-flash"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
