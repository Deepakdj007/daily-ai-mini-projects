import logging
from typing import Optional

from retell import AsyncRetell
from retell.types import RegisterCallResponse

from config import settings


logger = logging.getLogger(__name__)

_client: Optional[AsyncRetell] = None


def get_client() -> AsyncRetell:
    global _client
    if _client is None:
        _client = AsyncRetell(api_key=settings.RETELL_API_KEY)
    return _client


async def create_phone_call(
    to_number: str,
    metadata: Optional[dict] = None,
    dynamic_variables: Optional[dict] = None,
) -> RegisterCallResponse:
    """Initiate an outbound phone call via Retell."""
    if not settings.RETELL_FROM_NUMBER:
        raise ValueError("RETELL_FROM_NUMBER not set — run setup.py first")
    if not settings.RETELL_AGENT_ID:
        raise ValueError("RETELL_AGENT_ID not set — run setup.py first")

    client = get_client()
    try:
        response = await client.call.create_phone_call(
            from_number=settings.RETELL_FROM_NUMBER,
            to_number=to_number,
            override_agent_id=settings.RETELL_AGENT_ID,
            metadata=metadata or {},
            retell_llm_dynamic_variables=dynamic_variables or {},
        )
        logger.info("Retell call created: %s → %s", response.call_id, to_number)
        return response
    except Exception as e:
        logger.error("Retell create_phone_call failed: %s", e)
        raise


async def get_call(call_id: str):
    """Retrieve call details from Retell."""
    client = get_client()
    try:
        return await client.call.retrieve(call_id)
    except Exception as e:
        logger.error("Retell get_call failed for %s: %s", call_id, e)
        raise
