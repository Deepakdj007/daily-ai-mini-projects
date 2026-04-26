"""
One-time setup script.
Run once after putting RETELL_API_KEY in .env:

    uv run python setup.py

Creates a Retell agent + buys a US phone number, then writes
RETELL_AGENT_ID and RETELL_FROM_NUMBER back into ../.env (or .env).
"""
import asyncio
import logging
import os
import re
from pathlib import Path

from retell import AsyncRetell

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

AGENT_SYSTEM_PROMPT = """You are a friendly voice assistant that helps book medical appointments.
You speak both English and Malayalam fluently — respond in whichever language the caller uses.
Keep responses short (1-2 sentences). Be warm and professional.

Your goal is to collect:
1. The caller's name
2. Preferred appointment date and time
3. Reason for the visit (chief complaint)
4. Confirm the details back to them

Once confirmed, thank them and end the call."""

WEBHOOK_BASE_URL = os.getenv("WEBHOOK_BASE_URL", "http://localhost:8000")


def find_env_file() -> Path:
    """Look for .env in current dir or parent dir."""
    for p in [Path(".env"), Path("../.env")]:
        if p.exists():
            return p.resolve()
    # default: create in project root
    return (Path("..") / ".env").resolve()


def patch_env(path: Path, updates: dict[str, str]) -> None:
    """Write or update key=value lines in an .env file."""
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    for key, value in updates.items():
        pattern = rf"^{re.escape(key)}=.*$"
        line = f"{key}={value}"
        if re.search(pattern, text, flags=re.MULTILINE):
            text = re.sub(pattern, line, text, flags=re.MULTILINE)
        else:
            text = text.rstrip("\n") + f"\n{line}\n"
    path.write_text(text, encoding="utf-8")
    logger.info("Updated %s", path)


async def main() -> None:
    api_key = os.getenv("RETELL_API_KEY")
    if not api_key:
        # try loading from .env manually (setup runs before config.py is imported)
        env_file = find_env_file()
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("RETELL_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break

    if not api_key:
        raise SystemExit("RETELL_API_KEY not found. Set it in .env first.")

    client = AsyncRetell(api_key=api_key)

    # ── 1. Create agent ──────────────────────────────────────────────────────
    logger.info("Creating Retell agent…")
    agent = await client.agent.create(
        agent_name="DesiBot",
        response_engine={
            "type": "custom_llm",
            "llm_websocket_url": f"{WEBHOOK_BASE_URL}/retell/llm",
        },
        voice_id="11labs-Adrian",
        language="en-IN",
        responsiveness=1.0,
        interruption_sensitivity=1.0,
        enable_backchannel=True,
        ambient_sound="off",
        webhook_url=f"{WEBHOOK_BASE_URL}/webhooks/retell",
    )
    logger.info("Agent created: %s", agent.agent_id)

    # ── 2. Buy a US phone number ─────────────────────────────────────────────
    logger.info("Purchasing US phone number (area code 415)…")
    phone = await client.phone_number.create(
        area_code=415,
        outbound_agent_id=agent.agent_id,
    )
    logger.info("Number purchased: %s", phone.phone_number)

    # ── 3. Patch .env ────────────────────────────────────────────────────────
    env_path = find_env_file()
    patch_env(env_path, {
        "RETELL_AGENT_ID": agent.agent_id,
        "RETELL_FROM_NUMBER": phone.phone_number,
    })

    logger.info("\nSetup complete!")
    logger.info("  Agent ID  : %s", agent.agent_id)
    logger.info("  From Number: %s", phone.phone_number)
    logger.info("\nNow start the server:  uv run uvicorn main:app --reload --port 8000")


if __name__ == "__main__":
    asyncio.run(main())
