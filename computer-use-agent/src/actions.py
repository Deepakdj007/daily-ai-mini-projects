"""Act: execute a decided Action on the real screen with PyAutoGUI."""

import time

import pyautogui

from src.brain import Action
from src.config import WAIT_SECONDS


def execute_action(action: Action) -> str:
    """Execute one Action and return a short log string of what happened.

    Click coordinates arrive normalized 0-1000 and are mapped against the
    logical screen size, which is what pyautogui.click expects (DPI-correct).
    """
    if action.action == "click":
        width, height = pyautogui.size()
        x = int((action.x or 0) / 1000 * width)
        y = int((action.y or 0) / 1000 * height)
        pyautogui.click(x, y)
        return f"clicked ({x}, {y})"

    if action.action == "type":
        pyautogui.write(action.text or "", interval=0.02)
        return f"typed {action.text!r}"

    if action.action == "scroll":
        clicks = -500 if action.direction == "down" else 500
        pyautogui.scroll(clicks)
        return f"scrolled {action.direction}"

    if action.action == "wait":
        time.sleep(WAIT_SECONDS)
        return f"waited {WAIT_SECONDS}s"

    return "done"
