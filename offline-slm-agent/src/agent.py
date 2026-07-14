"""The hand-built agent loop: think -> call tools -> observe -> repeat -> answer.

One function, run_agent(), owns the whole ReAct cycle against a local Ollama model.
"""

from ollama import Client

from src.config import MAX_STEPS, MODEL, OLLAMA_HOST, SYSTEM_PROMPT
from src.tools import TOOL_FUNCTIONS, TOOL_MAP

# One client for the process. host=None falls back to http://localhost:11434.
_client = Client(host=OLLAMA_HOST)


def _execute_tool_call(name: str, arguments: dict) -> str:
    """Run one tool by name; return its result or a readable error for the model."""
    fn = TOOL_MAP.get(name)
    if fn is None:
        return f"Error: no tool named '{name}'. Available: {', '.join(TOOL_MAP)}."
    try:
        return fn(**arguments)
    except Exception as exc:  # feed failures back so the model can retry
        return f"Error running {name}: {exc}"


def run_agent(user_message: str) -> str:
    """Answer one user request, calling tools as many times as needed.

    Args:
      user_message: The user's task in plain English.
    Returns:
      The model's final text answer.
    """
    messages: list = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    for _ in range(MAX_STEPS):
        response = _client.chat(model=MODEL, messages=messages, tools=TOOL_FUNCTIONS)
        messages.append(response.message)

        # No tool calls means the model is done thinking — its content is the answer.
        if not response.message.tool_calls:
            return response.message.content or "(no answer)"

        for call in response.message.tool_calls:
            name = call.function.name
            arguments = dict(call.function.arguments)  # already a dict, not JSON text
            print(f"  -> {name}({arguments})")
            result = _execute_tool_call(name, arguments)
            print(f"  <- {result}")
            messages.append({"role": "tool", "tool_name": name, "content": str(result)})

    return "Stopped: hit the step limit before finishing. Try a simpler request."
