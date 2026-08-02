"""Entrypoint: wires Telegram handlers to the agent, opens the durable stores on
startup, and starts polling. Run with `PYTHONPATH=. uv run python -m src.main`.

Inputs:  TELEGRAM_BOT_TOKEN, GROQ_API_KEY, OWNER_CHAT_ID via src/config.py.
Outputs: a running bot process; owner-only replies in the configured Telegram chat.
"""

import asyncio
import logging

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from telegram import Bot, Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from . import config, readers, store
from .agent import ask, build_agent
from .google_tools import build_google_tools
from .scheduler import rehydrate_reminders, schedule_briefing, send_briefing
from .tools import build_tools

logger = logging.getLogger(__name__)


def _is_owner(update: Update) -> bool:
    """True once OWNER_CHAT_ID is set and this update came from that chat."""
    return bool(config.OWNER_CHAT_ID) and update.effective_chat.id == config.OWNER_CHAT_ID


async def _with_typing(bot: Bot, chat_id: int, coro):
    """Keep Telegram's "typing..." indicator alive while a slow coroutine runs.
    A free-tier Groq 429 retry can take 40-90 seconds, and Telegram's own
    indicator expires after ~5 seconds, so it has to be refreshed on a loop."""

    async def _keep_typing() -> None:
        while True:
            await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            await asyncio.sleep(4)

    typing_task = asyncio.create_task(_keep_typing())
    try:
        return await coro
    finally:
        typing_task.cancel()


async def on_startup(app: Application) -> None:
    """Open the checkpoint DB, build the agent + tools, and start the scheduler.
    Runs once run_polling() has an event loop, before it starts receiving updates."""
    conn = await aiosqlite.connect(str(config.MEM_PATH))
    app.bot_data["_mem_conn"] = conn  # closed in on_shutdown

    app.bot_data["db"] = store.connect(config.DB_PATH)
    llm = config.make_llm()
    app.bot_data["llm"] = llm

    tools = build_tools(app) + build_google_tools()
    app.bot_data["agent"] = build_agent(llm, tools, AsyncSqliteSaver(conn))

    schedule_briefing(app)
    rehydrate_reminders(app)


async def on_shutdown(app: Application) -> None:
    """Close the checkpoint connection — leaving it open can make the graph
    appear to hang on exit."""
    conn = app.bot_data.get("_mem_conn")
    if conn is not None:
        await conn.close()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Bootstrap: always reveals the chat id so the reader can set OWNER_CHAT_ID.
    Once that's set, only the owner gets a reply — everyone else stays silent."""
    chat_id = update.effective_chat.id
    if config.OWNER_CHAT_ID and chat_id != config.OWNER_CHAT_ID:
        return
    if config.OWNER_CHAT_ID:
        await update.message.reply_text(f"Hey, it's me. Your chat id is {chat_id}. Try /help.")
    else:
        await update.message.reply_text(
            f"Your chat id is {chat_id}.\n\n"
            f"Add this to .env and restart me:\nOWNER_CHAT_ID={chat_id}"
        )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_owner(update):
        return
    await update.message.reply_text(
        "I'm your personal assistant. Send me anything — I can search the web, "
        "remember notes, set reminders, and summarize links or PDFs you send me.\n"
        "/briefing fires the morning digest right now, instead of waiting for it."
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_owner(update):
        return
    agent = context.application.bot_data["agent"]
    chat_id = update.effective_chat.id
    reply = await _with_typing(context.bot, chat_id, ask(agent, chat_id, update.message.text))
    await update.message.reply_text(reply)


async def handle_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_owner(update):
        return
    document = update.message.document
    if document.file_size and document.file_size > config.MAX_DOCUMENT_BYTES:
        await update.message.reply_text("That PDF is over 20 MB — Telegram won't let me download it.")
        return

    telegram_file = await document.get_file()
    data = await telegram_file.download_as_bytearray()
    try:
        text = readers.pdf_text(bytes(data), config.MAX_TOOL_CHARS)
    except ValueError as exc:
        await update.message.reply_text(str(exc))
        return

    agent = context.application.bot_data["agent"]
    chat_id = update.effective_chat.id
    reply = await _with_typing(
        context.bot, chat_id, ask(agent, chat_id, f"Summarize this PDF:\n\n{text}")
    )
    await update.message.reply_text(reply)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("apscheduler").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    config.require_keys()

    app = (
        ApplicationBuilder()
        .token(config.TELEGRAM_BOT_TOKEN)
        .post_init(on_startup)
        .post_shutdown(on_shutdown)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("briefing", send_briefing))
    app.add_handler(MessageHandler(filters.Document.PDF, handle_pdf))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    app.run_polling()


if __name__ == "__main__":
    main()
