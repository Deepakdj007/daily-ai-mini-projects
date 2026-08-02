"""The two things that make this feel like an agent instead of a chatbot: reminders
that fire on their own, and a morning briefing nobody had to ask for.

Inputs:  a running telegram.ext.Application (JobQueue, bot, and bot_data["db"]/["llm"]).
Outputs: schedules JobQueue jobs; send_briefing also doubles as the /briefing handler.
"""

from datetime import datetime
from datetime import time as dt_time
from zoneinfo import ZoneInfo

from ddgs import DDGS
from telegram.ext import Application, ContextTypes

from . import config, store
from .prompts import BRIEFING_PROMPT


def _parse_due_at(due_at: str) -> datetime:
    """Parse an ISO datetime string, assuming the local timezone if none is given."""
    parsed = datetime.fromisoformat(due_at)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo(config.TIMEZONE))
    return parsed


async def _fire_reminder(context: ContextTypes.DEFAULT_TYPE) -> None:
    """JobQueue callback: mark the reminder fired and ping the owner."""
    data = context.job.data
    conn = context.application.bot_data["db"]
    store.mark_fired(conn, data["reminder_id"])
    await context.bot.send_message(chat_id=data["chat_id"], text=f"Reminder: {data['text']}")


def schedule_reminder(app: Application, reminder_id: int, chat_id: int, text: str, due_at: str) -> None:
    """Schedule a one-off ping. Using a delay-in-seconds (not an absolute time)
    means an already-overdue due_at just fires almost immediately — which is
    exactly what rehydrate_reminders relies on after a restart."""
    due = _parse_due_at(due_at)
    delay = max((due - datetime.now(ZoneInfo(config.TIMEZONE))).total_seconds(), 0)
    app.job_queue.run_once(
        _fire_reminder,
        when=delay,
        data={"reminder_id": reminder_id, "chat_id": chat_id, "text": text},
        name=f"reminder-{reminder_id}",
    )


def rehydrate_reminders(app: Application) -> None:
    """On startup, re-schedule every reminder SQLite still says is pending.
    JobQueue jobs live only in memory — SQLite is the durable source of truth,
    so a reminder set before the bot was killed still fires after it restarts."""
    conn = app.bot_data["db"]
    for row in store.all_pending_reminders(conn):
        schedule_reminder(app, row["id"], row["chat_id"], row["text"], row["due_at"])


def schedule_briefing(app: Application) -> None:
    """Register the daily morning briefing at config.BRIEFING_HOUR, local time."""
    app.job_queue.run_daily(
        send_briefing,
        time=dt_time(hour=config.BRIEFING_HOUR, tzinfo=ZoneInfo(config.TIMEZONE)),
        name="morning-briefing",
    )


async def send_briefing(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Assemble today's reminders, fresh headlines, and recent notes into one
    message and send it unprompted. Also used directly as the /briefing handler
    so the daily job can be tested without waiting for the actual hour."""
    conn = context.application.bot_data["db"]
    reminders = store.pending_reminders(conn, config.OWNER_CHAT_ID)
    notes = store.recent_notes(conn)

    headline_blocks = []
    for topic in config.BRIEFING_TOPICS:
        try:
            results = DDGS().text(topic, max_results=3)
        except Exception:
            results = []
        bullets = "\n".join(f"- {r['title']}" for r in results) or "- (search unavailable)"
        headline_blocks.append(f"{topic}:\n{bullets}")

    prompt = BRIEFING_PROMPT.format(
        reminders="\n".join(f"- {r['text']} at {r['due_at']}" for r in reminders) or "None today.",
        notes="\n".join(f"- {n['text']}" for n in notes) or "None recent.",
        headlines="\n\n".join(headline_blocks),
    )
    llm = context.application.bot_data["llm"]
    reply = await llm.ainvoke(prompt)
    await context.bot.send_message(chat_id=config.OWNER_CHAT_ID, text=reply.content)
