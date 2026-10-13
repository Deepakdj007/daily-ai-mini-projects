"""Three adapters behind one interface: `await provider.complete(messages)`.

Inputs:  chat history in one common shape - [{"role": "system"|"user"|"assistant",
         "content": str}, ...] - the same list no matter who answers.
Outputs: the reply text as a str, or a ProviderError (see errors.py).
"""

import asyncio
from collections.abc import Callable

import ollama
from google import genai
from google.genai import types
from groq import AsyncGroq

from src import config
from src.errors import to_provider_error

Messages = list[dict[str, str]]


class Provider:
    """Base adapter: timeout, chaos hook and error translation live here once."""

    name = "base"

    def __init__(self) -> None:
        self.chaos: Callable[[str], None] | None = None  # set by chaos.py for demos

    async def complete(self, messages: Messages) -> str:
        """Ask the model for a JSON reply. Raises ProviderError on any failure."""
        try:
            if self.chaos:
                self.chaos(self.name)
            return await asyncio.wait_for(self._call(messages), config.CALL_TIMEOUT_S)
        except Exception as exc:
            raise to_provider_error(self.name, exc) from exc

    async def _call(self, messages: Messages) -> str:
        """The one SDK call each subclass implements."""
        raise NotImplementedError


class GeminiProvider(Provider):
    """Primary: Gemini via google-genai. It has no 'system' or 'assistant' roles."""

    name = "gemini"

    def __init__(self, api_key: str) -> None:
        super().__init__()
        # No key = no client. The provider then fails like a dead one instead of crashing
        # startup. No retry_options passed, so the SDK never retries behind our back.
        self.client = genai.Client(api_key=api_key) if api_key else None

    async def _call(self, messages: Messages) -> str:
        """Move the system prompt into config and rename 'assistant' to 'model'."""
        if self.client is None:
            raise ValueError("GEMINI_API_KEY is not set")
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [
            types.Content(role="model" if m["role"] == "assistant" else "user",
                          parts=[types.Part(text=m["content"])])
            for m in messages if m["role"] != "system"
        ]
        response = await self.client.aio.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        return response.text or ""


class GroqProvider(Provider):
    """Fallback: Groq speaks the OpenAI message format, so messages pass straight through."""

    name = "groq"

    def __init__(self, api_key: str) -> None:
        super().__init__()
        # max_retries=0: AsyncGroq retries twice by default, and our router owns retries.
        self.client = AsyncGroq(api_key=api_key, max_retries=0) if api_key else None

    async def _call(self, messages: Messages) -> str:
        """One chat completion in JSON mode."""
        if self.client is None:
            raise ValueError("GROQ_API_KEY is not set")
        response = await self.client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or ""


class OllamaProvider(Provider):
    """Last resort: a small model on this machine. No key, no quota, no internet."""

    name = "ollama"

    def __init__(self) -> None:
        super().__init__()
        self.client = ollama.AsyncClient()

    async def _call(self, messages: Messages) -> str:
        """One local chat call with JSON output forced."""
        response = await self.client.chat(model=config.OLLAMA_MODEL, messages=messages, format="json")
        return response.message.content or ""


def build_providers(gemini_key: str = config.GEMINI_API_KEY,
                    groq_key: str = config.GROQ_API_KEY) -> list[Provider]:
    """Create the providers in failover order (see config.PROVIDER_ORDER)."""
    by_name: dict[str, Provider] = {
        "gemini": GeminiProvider(gemini_key),
        "groq": GroqProvider(groq_key),
        "ollama": OllamaProvider(),
    }
    return [by_name[name] for name in config.PROVIDER_ORDER]
