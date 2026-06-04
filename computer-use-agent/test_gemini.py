"""Throwaway check that the Gemini key works with google-genai.

Run: PYTHONPATH=. uv run python test_gemini.py
Expected: prints a short confirmation line from the model.
"""

from dotenv import load_dotenv
from google import genai

# Must run before genai.Client() — the client reads GEMINI_API_KEY / GOOGLE_API_KEY
# from the environment, and uv run does not auto-load .env on Windows.
load_dotenv()

client = genai.Client()

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents="Reply with exactly: Gemini is working",
)
print(response.text)
