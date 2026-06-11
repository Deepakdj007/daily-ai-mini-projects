# live-vision-agent

A real-time AI agent that watches through your webcam and talks with you over your speakers, built on the Gemini Live API (`gemini-3.1-flash-live-preview`, free tier). Your microphone and camera stream continuously to the model over one WebSocket session; it answers in natural speech with a live transcript in the terminal, and you can interrupt it mid-sentence just by speaking.

## Run it

```
cp .env.example .env        # paste your key from https://aistudio.google.com/apikey
PYTHONPATH=. uv run python src/agent.py
```

Use headphones — on open speakers the agent hears itself talk.
