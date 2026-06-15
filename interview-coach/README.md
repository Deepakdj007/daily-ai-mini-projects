# interview-coach

Real-time AI interview coach powered by Gemini Live. Watches you through the webcam and listens as you answer mock interview questions — coaches you on tone, pacing, filler words, body language, and content after each answer.

## Setup

```bash
cp .env.example .env
# paste your key from https://aistudio.google.com/apikey
```

## Run

```bash
PYTHONPATH=. uv run python src/coach.py
```

Press `Ctrl+C` to end the session and see your summary report.
