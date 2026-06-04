"""Step 5 — Think: decide the next action and PREVIEW the target safely.

This does NOT click. For a 'click' action it moves the mouse to where it would
click so you can confirm the targeting (and DPI mapping) are correct.

Run: PYTHONPATH=. uv run python think_test.py
"""

import pyautogui
from dotenv import load_dotenv
from google import genai

from src.brain import decide_action
from src.screen import capture_screen

load_dotenv()
client = genai.Client()

TASK = "Open the File menu in the current application"

screenshot = capture_screen()
action = decide_action(client, TASK, screenshot)
print("Decided action:", action.model_dump())

if action.action == "click" and action.x is not None and action.y is not None:
    width, height = pyautogui.size()  # logical screen size — DPI-correct for clicks
    x = int(action.x / 1000 * width)
    y = int(action.y / 1000 * height)
    print(f"Would click screen pixel ({x}, {y}). Moving mouse there now (no click).")
    pyautogui.moveTo(x, y, duration=1.0)
    print("Mouse moved. Check where the pointer landed.")
