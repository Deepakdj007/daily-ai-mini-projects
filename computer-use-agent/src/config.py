"""Configuration, constants, and safety settings for the computer-use agent."""

import pyautogui

# Gemini model used for both seeing the screen and deciding the next action.
MODEL = "gemini-3.5-flash"

# Hard cap on loop iterations so a confused agent cannot run away.
MAX_STEPS = 15

# Seconds the "wait" action sleeps, to let an app or menu finish opening.
WAIT_SECONDS = 1.5

# Downscale screenshots to this width before sending to Gemini (smaller = faster
# and cheaper). Gemini returns coordinates normalized to 0-1000, so resizing the
# image here does NOT affect how we map clicks back to the real screen.
SCREENSHOT_WIDTH = 1280

# --- PyAutoGUI safety (applied process-wide at import) ---
# Emergency brake: slam the mouse pointer into a screen corner to abort instantly
# (PyAutoGUI raises FailSafeException). Keep a corner reachable while testing.
pyautogui.FAILSAFE = True

# Pause after every PyAutoGUI action so the UI can react and you can watch/abort.
pyautogui.PAUSE = 0.5
