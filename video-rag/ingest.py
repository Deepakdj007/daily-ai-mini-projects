# ingest.py
import os
import time
from pathlib import Path

import cv2
import yt_dlp
from google import genai
from google.genai import types

import config

os.environ["CHROMA_TELEMETRY"] = "false"
import chromadb

def download_video(url: str) -> Path:
    """Download a YouTube video and return the local file path."""
    config.VIDEO_DIR.mkdir(exist_ok=True)

    ydl_opts = {
        "format": config.VIDEO_FORMAT,
        "outtmpl": str(config.VIDEO_DIR / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_path = config.VIDEO_DIR / f"{info['id']}.{info.get('ext', 'mp4')}"

    return video_path

def extract_frames(video_path: Path) -> list[dict]:
    """Extract one JPEG frame every FRAME_INTERVAL_SECONDS from the video."""
    config.FRAMES_DIR.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    print(f"  Video: {duration:.0f}s at {fps:.1f} fps")

    frame_interval = max(1, int(fps * config.FRAME_INTERVAL_SECONDS))
    extracted = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / fps
            frame_path = config.FRAMES_DIR / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(
                str(frame_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY],
            )
            extracted.append({
                "frame_index": frame_idx,
                "timestamp": timestamp,
                "path": frame_path,
            })

        frame_idx += 1

    cap.release()
    print(f"  Extracted {len(extracted)} frames")
    return extracted

def describe_frame(client: genai.Client, frame_path: Path) -> str:
    """Send a frame to Gemini Vision and get a detailed text description."""
    with open(frame_path, "rb") as f:
        img_bytes = f.read()

    prompt = (
        "Describe this video frame in detail. Include: "
        "any text visible on screen such as whiteboards, slides, terminals, or code, "
        "any diagrams, charts, or visual elements, "
        "what is happening in the scene, "
        "and any tools or interfaces visible. "
        "Be specific and thorough. Write 3-5 sentences."
    )

    response = client.models.generate_content(
        model=config.VISION_MODEL,
        contents=[
            types.Part.from_text(text=prompt),
            types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
        ],
    )

    return response.text.strip()

def embed_text(client: genai.Client, text: str) -> list[float]:
    """Convert a text description into a 3072-dimensional vector."""
    response = client.models.embed_content(
        model=config.EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=config.EMBED_DIM,
        ),
    )
    return response.embeddings[0].values

def _format_timestamp(seconds: float) -> str:
    """Convert a float number of seconds to MM:SS display format."""
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes:02d}:{secs:02d}"

def ingest(url: str) -> tuple[genai.Client, chromadb.Collection]:
    """
    Full ingestion pipeline: download → extract → describe → embed → store.

    Returns (genai_client, chroma_collection) so cli.py can reuse the same
    Gemini client for querying instead of creating a second one.
    """
    print(f"  Using API key from .env: ...{config.GEMINI_API_KEY[-6:]}")
    client = genai.Client(api_key=config.GEMINI_API_KEY)

    # ── Step 1: Download ──────────────────────────────────────────────────────
    print("⬇️  Downloading video...")
    video_path = download_video(url)
    print(f"  Saved to {video_path}")

    # ── Step 2: Extract frames ────────────────────────────────────────────────
    print(f"\n🎞️  Extracting frames (1 per {config.FRAME_INTERVAL_SECONDS}s)...")
    frames = extract_frames(video_path)

    # ── Steps 3, 4, 5: Describe → Embed → Store (one frame at a time) ────────
    print("\n🔍 Describing and indexing frames...")

    db = chromadb.Client()
    collection = db.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    for i, frame in enumerate(frames):
        timestamp_str = _format_timestamp(frame["timestamp"])
        print(f"  [{i+1}/{len(frames)}] t={timestamp_str} — describing...", end="", flush=True)

        description = describe_frame(client, frame["path"])
        vector = embed_text(client, description)

        collection.add(
            ids=[f"frame_{frame['frame_index']:06d}"],
            embeddings=[vector],
            documents=[description],
            metadatas=[{
                "timestamp": frame["timestamp"],
                "timestamp_str": timestamp_str,
                "frame_path": str(frame["path"]),
                "frame_index": frame["frame_index"],
            }],
        )

        print(" done.")

        # Free tier limit: 15 RPM across all Gemini calls.
        # Each frame costs 2 calls (describe + embed).
        # Sleeping 2s per frame keeps us well under the limit.
        if i < len(frames) - 1:
            time.sleep(2)

    print(f"\n✅ Indexed {len(frames)} frames into ChromaDB")
    return client, collection

