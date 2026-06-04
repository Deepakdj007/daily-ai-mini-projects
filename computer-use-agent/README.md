# computer-use-agent

An AI agent that controls your computer — mouse, keyboard, screenshots — to carry out a task you describe in plain English. It runs a See → Think → Act loop: PyAutoGUI captures the screen, Gemini Flash (via `google-genai`) decides the next action with click coordinates, PyAutoGUI executes it, and LangGraph repeats until the task is done or a step cap is reached.

## Run

Set `GEMINI_API_KEY` in `.env` (see `.env.example`). Run a task on your **primary monitor** (PyAutoGUI only sees and clicks the primary display):

```bash
PYTHONPATH=. uv run python main.py "open notepad, then type: hello from my AI agent"
```

Safety: move the mouse to a screen corner at any time to abort (PyAutoGUI failsafe). The loop is capped at 15 steps.
