# agent/llm.py
import json
from agent.client import client


def call_and_parse(messages: list[dict]) -> dict:
    """
    Send the multimodal messages to Llama 4 Scout via Groq.
    Parse the JSON response and return it as a Python dict.

    On JSON parse failure, returns a structured error dict
    instead of raising — so the caller always gets a dict back.
    """

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=messages,
        response_format={"type": "json_object"},  # JSON mode: forces valid JSON output
        temperature=0.2,   # low temperature = consistent, precise extraction
        max_tokens=1024,
    )

    raw = response.choices[0].message.content

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        # Return a structured error dict instead of crashing
        # The UI can display this gracefully
        return {
            "error": "Model returned invalid JSON",
            "detail": str(e),
            "raw_output": raw,
        }