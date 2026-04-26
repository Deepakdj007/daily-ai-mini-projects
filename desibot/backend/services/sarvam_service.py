import logging
from typing import Optional

from sarvamai import SarvamAI

from config import settings


logger = logging.getLogger(__name__)

_client: Optional[SarvamAI] = None


def get_client() -> SarvamAI:
    global _client
    if _client is None:
        _client = SarvamAI(api_subscription_key=settings.SARVAM_API_KEY)
    return _client


async def chat_completion(messages: list[dict]) -> str:
    """Call Sarvam-M (free) and return the assistant reply text."""
    client = get_client()
    try:
        response = client.chat.completions(messages=messages)
        return response.choices[0].message.content
    except Exception as e:
        logger.error("Sarvam chat completion failed: %s", e)
        raise
