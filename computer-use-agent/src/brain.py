"""Think: ask Gemini to choose the single next action from the screen + task.

Gemini returns a structured Action (not prose) via response_schema, so the
rest of the app can execute it reliably.
"""

from typing import Literal, Optional

from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel

from src.config import MODEL


class Action(BaseModel):
    """One step the agent wants to take. Coordinates are normalized 0-1000."""

    reasoning: str
    action: Literal["click", "type", "scroll", "wait", "done"]
    x: Optional[int] = None          # click target X, normalized 0-1000
    y: Optional[int] = None          # click target Y, normalized 0-1000
    text: Optional[str] = None       # text to type (for "type")
    direction: Optional[Literal["up", "down"]] = None  # scroll direction


SYSTEM_INSTRUCTION = (
    "You control a Windows computer to accomplish the user's task. You are given "
    "the task and a screenshot of the current screen. Decide the SINGLE next "
    "action that makes progress.\n"
    "Coordinates x and y are normalized 0-1000 (0,0 = top-left, 1000,1000 = "
    "bottom-right of the screenshot).\n"
    "Actions:\n"
    "- click: set x,y to the CENTER of the target element.\n"
    "- type: set text to the string to type (the field must already be focused "
    "from a previous click).\n"
    "- scroll: set direction to up or down.\n"
    "- wait: pause briefly to let an app or menu finish opening, then look again.\n"
    "- done: the task is fully complete.\n"
    "Windows tips:\n"
    "- To open an app: click the taskbar Search icon, type the app name, then "
    "click the top result. Prefer TYPING the name over hunting for 'Recent' or "
    "pinned tiles.\n"
    "- If the Search box or Start menu is ALREADY open, do NOT click the search "
    "icon again — just type.\n"
    "- Right after opening an app or triggering search, if it has not appeared "
    "yet, use 'wait' once, then re-check.\n"
    "You are given the actions you have ALREADY taken. Do NOT repeat an action "
    "that already worked — build on it. If the screen shows your last action had "
    "no visible effect, try a DIFFERENT approach. When the task is visibly "
    "accomplished on screen, return the 'done' action.\n"
    "Always explain your choice in 'reasoning'. Return exactly one action."
)


def decide_action(
    client: genai.Client,
    task: str,
    screenshot: Image.Image,
    history: list[str],
) -> Action:
    """Send task + history + screenshot to Gemini and get back one Action."""
    history_text = "\n".join(f"- {h}" for h in history) or "none yet"
    response = client.models.generate_content(
        model=MODEL,
        contents=[
            f"Task: {task}",
            f"Actions you have already taken (most recent last):\n{history_text}",
            "Here is the current screen. Decide the next action:",
            screenshot,
        ],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=Action,
        ),
    )
    return response.parsed
