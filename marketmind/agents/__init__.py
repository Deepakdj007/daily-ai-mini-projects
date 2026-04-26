# agents/__init__.py
import os
import json
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Shared Groq client — used by all agents
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1,   # low = consistent, structured outputs
    max_tokens=500,
)


def safe_parse_json(content: str) -> dict:
    """Parse LLM JSON response, handling markdown code fences.

    Llama models sometimes wrap JSON in ```json ... ``` even when asked
    not to. This function strips those fences before parsing.
    """
    content = content.strip()
    if content.startswith("```"):
        for part in content.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                content = part
                break
    return json.loads(content.strip())