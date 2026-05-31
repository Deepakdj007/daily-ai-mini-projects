"""Throwaway check that GROQ_API_KEY works with the async Groq client.

Run: PYTHONPATH=. uv run python test_groq.py
Expected: prints a short confirmation line from the model.
"""

import asyncio

from dotenv import load_dotenv
from groq import AsyncGroq

# Must run before AsyncGroq() — the client reads GROQ_API_KEY from the
# environment at construction time, and uv run does not auto-load .env on Windows.
load_dotenv()


async def main() -> None:
    """Make one tiny chat completion call and print the result."""
    client = AsyncGroq()
    response = await client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": "Reply with exactly: Groq is working"},
        ],
        max_tokens=20,
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    asyncio.run(main())
