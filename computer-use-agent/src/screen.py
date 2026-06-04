"""Screen capture: grab the current screen and prepare it for the vision model."""

import pyautogui
from PIL import Image

from src.config import SCREENSHOT_WIDTH


def capture_screen() -> Image.Image:
    """Capture the full screen and downscale it for the vision model.

    Returns a PIL Image. Clicks are mapped against the real screen size later,
    so downscaling here only reduces tokens — it does not hurt click accuracy.
    """
    screenshot = pyautogui.screenshot()  # PIL Image of the whole screen
    if screenshot.width > SCREENSHOT_WIDTH:
        ratio = SCREENSHOT_WIDTH / screenshot.width
        new_height = int(screenshot.height * ratio)
        screenshot = screenshot.resize((SCREENSHOT_WIDTH, new_height))
    return screenshot
