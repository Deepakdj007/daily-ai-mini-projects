"""Step 4 — verify the 'See' stage: capture the screen and have Gemini describe it.

Run: PYTHONPATH=. uv run python see_test.py
Expected: a 2-3 sentence description of whatever is currently on your screen.
"""

from dotenv import load_dotenv
from google import genai

from src.config import MODEL
from src.screen import capture_screen

load_dotenv()

client = genai.Client()
image = capture_screen()

response = client.models.generate_content(
    model=MODEL,
    contents=["Describe what is on this screen in 2-3 sentences.", image],
)
print(response.text)
