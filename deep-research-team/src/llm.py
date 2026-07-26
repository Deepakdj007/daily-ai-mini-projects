"""Two thin wrappers around the Gemini client.

`astructured` is used wherever a step needs a typed answer it can branch on --
the planner's question list and the reflector's verdict. LlamaIndex builds the
JSON schema from the pydantic model, so there is no hand-written schema and no
parsing of fenced code blocks.

`atext` is used where free prose is the point: the per-question summaries and
the final report.

Inputs:  a GoogleGenAI client, a prompt, and (for astructured) a pydantic model.
Outputs: a validated model instance, or a plain string.
"""

from typing import TypeVar

from llama_index.llms.google_genai import GoogleGenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


async def astructured(llm: GoogleGenAI, prompt: str, schema: type[T]) -> T:
    """Ask the model for an answer shaped like `schema`.

    Args:
        llm: a client from config.make_llm().
        prompt: the instruction to send.
        schema: the pydantic model the answer must match.

    Returns:
        An instance of `schema`, already validated.
    """
    structured = llm.as_structured_llm(schema)
    response = await structured.acomplete(prompt)
    # .raw holds the parsed pydantic object; str(response) would give back JSON text.
    return response.raw


async def atext(llm: GoogleGenAI, prompt: str) -> str:
    """Ask the model for plain text.

    Args:
        llm: a client from config.make_llm().
        prompt: the instruction to send.

    Returns:
        The model's reply, stripped.
    """
    response = await llm.acomplete(prompt)
    return str(response).strip()
