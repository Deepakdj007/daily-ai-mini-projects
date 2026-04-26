# agent/agent.py
from agent.image import load_from_path, load_from_bytes
from agent.prompt import build_messages
from agent.llm import call_and_parse


def run_from_path(image_path: str, instruction: str, context: str = "") -> dict:
    """
    Run the agent with an image file path.
    Use this for scripts, tests, and programmatic use.
    """
    base64_image, media_type = load_from_path(image_path)
    messages = build_messages(base64_image, media_type, instruction, context)
    return call_and_parse(messages)


def run_from_bytes(
    image_bytes: bytes,
    filename: str,
    instruction: str,
    context: str = "",
) -> dict:
    """
    Run the agent with raw image bytes.
    Use this from the Streamlit UI — Streamlit gives us bytes, not a file path.
    """
    base64_image, media_type = load_from_bytes(image_bytes, filename)
    messages = build_messages(base64_image, media_type, instruction, context)
    return call_and_parse(messages)