# config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)
os.environ.pop("GOOGLE_API_KEY", None)

# ── API ───────────────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.environ["GEMINI_API_KEY"]

# ── Models ───────────────────────────────────────────────────────────────────
VISION_MODEL: str = "gemini-2.5-flash"
EMBED_MODEL: str = "gemini-embedding-001"
EMBED_DIM: int = 3072

# ── Frame extraction ──────────────────────────────────────────────────────────
FRAME_INTERVAL_SECONDS: int = 30  # one frame every 30 seconds
JPEG_QUALITY: int = 85            # 85 is a good balance of size vs. quality
VIDEO_FORMAT: str = "bestvideo[height<=720][ext=mp4]/best[height<=720]"

# ── Vector store ──────────────────────────────────────────────────────────────
COLLECTION_NAME: str = "video_frames"
TOP_K_RESULTS: int = 3            # how many frames to retrieve per question

# ── Paths ─────────────────────────────────────────────────────────────────────
FRAMES_DIR: Path = Path("frames")
VIDEO_DIR: Path = Path("videos")