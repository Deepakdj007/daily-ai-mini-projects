"""The tools the agent routes to. Each @tool docstring is what the model reads to
decide when to call it, so the docstrings are written for the model, not the reader.

Inputs:  a running telegram.ext.Application (for the JobQueue and the notes/reminders DB).
Outputs: build_tools(app) returns the list passed to create_agent().

This is a single-owner assistant (see the chat-id gate in main.py), so reminders are
always scoped to config.OWNER_CHAT_ID rather than threading a chat id through every tool.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

from ddgs import DDGS
from langchain_core.tools import tool
from telegram.ext import Application

from . import config, readers, store
from .scheduler import schedule_reminder


def build_tools(app: Application) -> list:
    """Close over the running Application so tools can reach the DB and JobQueue."""
    conn = app.bot_data["db"]

    @tool
    def current_datetime() -> str:
        """Return the current date and time in the user's local timezone as an ISO
        8601 string. Always call this before resolving a relative date or time
        such as "tomorrow", "in 3 minutes", or "next Monday" — never guess."""
        return datetime.now(ZoneInfo(config.TIMEZONE)).isoformat(timespec="seconds")

    @tool
    def web_search(query: str) -> str:
        """Search the web for fresh or current information not in your training
        data — news, prices, scores, anything time-sensitive. Returns the top
        results as title, link, and snippet."""
        try:
            results = DDGS().text(query, max_results=config.SEARCH_MAX_RESULTS)
        except Exception as exc:
            return f"Web search failed: {exc}"
        if not results:
            return "No results found."
        # ddgs's "auto" backend can scrape full pages for some results, so a
        # "snippet" can come back much larger than expected — clamp it like
        # every other tool output before it reaches the model.
        lines = [f"- {r['title']}: {r['body']} ({r['href']})" for r in results]
        return readers.truncate("\n".join(lines), config.MAX_TOOL_CHARS)

    @tool
    def save_note(text: str, tags: str = "") -> str:
        """Save a short personal note for later recall, e.g. "my wifi password is
        X" or "dentist appointment card is in the drawer". tags is an optional
        comma-separated string for topics like "passwords,home"."""
        note_id = store.add_note(conn, text, tags)
        return f"Saved note #{note_id}."

    @tool
    def search_notes(query: str) -> str:
        """Search previously saved personal notes by keyword. Always try this
        before web_search when the question is about the user's own life, not
        the wider world."""
        rows = store.search_notes(conn, query)
        if not rows:
            return "No matching notes."
        return "\n".join(f"- ({row['created_at']}) {row['text']}" for row in rows)

    @tool
    def add_reminder(text: str, due_at: str) -> str:
        """Schedule a reminder. due_at must be a full ISO 8601 datetime string
        with the user's local timezone offset, e.g. "2026-07-29T15:04:00+05:30" —
        compute it from current_datetime plus the requested offset, never guess
        the current date."""
        reminder_id = store.add_reminder(conn, config.OWNER_CHAT_ID, text, due_at)
        schedule_reminder(app, reminder_id, config.OWNER_CHAT_ID, text, due_at)
        return f"Reminder #{reminder_id} set for {due_at}."

    @tool
    def list_reminders() -> str:
        """List all pending (not yet fired or cancelled) reminders, soonest first."""
        rows = store.pending_reminders(conn, config.OWNER_CHAT_ID)
        if not rows:
            return "No pending reminders."
        return "\n".join(f"- #{row['id']} at {row['due_at']}: {row['text']}" for row in rows)

    @tool
    def cancel_reminder(reminder_id: int) -> str:
        """Cancel a pending reminder by its id, as shown by list_reminders."""
        ok = store.cancel_reminder(conn, config.OWNER_CHAT_ID, reminder_id)
        return f"Cancelled reminder #{reminder_id}." if ok else (
            f"No pending reminder #{reminder_id} found."
        )

    @tool
    def read_link(url: str) -> str:
        """Fetch a web article or YouTube video URL and return its text (or
        transcript) so you can summarize or answer questions about it."""
        try:
            video_id = readers.youtube_video_id(url)
            if video_id:
                return readers.youtube_text(video_id, config.MAX_TOOL_CHARS)
            return readers.article_text(url, config.MAX_TOOL_CHARS)
        except Exception as exc:
            return f"Could not read that link: {exc}"

    return [
        current_datetime,
        web_search,
        save_note,
        search_notes,
        add_reminder,
        list_reminders,
        cancel_reminder,
        read_link,
    ]
