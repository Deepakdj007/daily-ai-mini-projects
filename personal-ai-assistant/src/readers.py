"""Turn a URL or a PDF's bytes into plain text the agent can summarize.

Inputs:  a URL string, or raw PDF bytes.
Outputs: plain text, already truncated to MAX_TOOL_CHARS so a single long page or
         transcript can't burn a large share of Groq's daily free token cap.

These are pure functions with no LLM calls — tools.py wraps them and turns
failures into a friendly message instead of a traceback.
"""

import re
from io import BytesIO

from pypdf import PdfReader
from trafilatura import extract, fetch_url
from youtube_transcript_api import YouTubeTranscriptApi

_YOUTUBE_ID_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?v=|shorts/)|youtu\.be/)([A-Za-z0-9_-]{11})"
)


def youtube_video_id(url: str) -> str | None:
    """Return the 11-character video id if url is a YouTube link, else None."""
    match = _YOUTUBE_ID_RE.search(url)
    return match.group(1) if match else None


def truncate(text: str, limit: int) -> str:
    """Clamp text to limit characters, noting when it was cut."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n[...truncated, {len(text) - limit} more characters omitted]"


def youtube_text(video_id: str, limit: int) -> str:
    """Fetch a YouTube video's transcript and return it as plain text.

    Raises whatever youtube_transcript_api raises (e.g. transcripts disabled) —
    the caller is expected to catch and report it.
    """
    fetched = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
    text = " ".join(snippet.text for snippet in fetched)
    return truncate(text, limit)


def article_text(url: str, limit: int) -> str:
    """Download a web page and extract its main readable text.

    Raises ValueError if the page could not be downloaded or had no extractable text.
    """
    downloaded = fetch_url(url)
    if not downloaded:
        raise ValueError(f"could not download {url}")
    text = extract(downloaded, include_comments=False, include_tables=False)
    if not text:
        raise ValueError(f"no readable text found at {url}")
    return truncate(text, limit)


def pdf_text(data: bytes, limit: int) -> str:
    """Extract text from PDF bytes. Scanned/image-only pages contribute nothing —
    pypdf does no OCR, so extract_text() can return None per page."""
    reader = PdfReader(BytesIO(data))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    if not text.strip():
        raise ValueError("this PDF has no extractable text (likely a scanned image)")
    return truncate(text, limit)
