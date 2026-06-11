"""Build the smolagents CodeAgent powered by Groq.

The agent writes Python code as its actions, runs it locally, reads the result,
and loops until it calls final_answer. It can search the web and call our custom
prime_factors tool, and its generated code may import the allowlisted modules.
"""

from smolagents import CodeAgent, LiteLLMModel, WebSearchTool

from src.config import AUTHORIZED_IMPORTS, MAX_STEPS, MODEL, get_api_key
from src.tools import prime_factors


def build_agent() -> CodeAgent:
    """Construct a ready-to-run CodeAgent with the model, tools, and limits."""
    model = LiteLLMModel(model_id=MODEL, api_key=get_api_key())

    # WebSearchTool (DuckDuckGo) covers live-info problems; prime_factors shows
    # how a custom tool slots in next to a built-in one.
    tools = [WebSearchTool(), prime_factors]

    return CodeAgent(
        tools=tools,
        model=model,
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        max_steps=MAX_STEPS,
        # verbosity_level=2 prints each code block the agent writes and its
        # execution output — that visible write-run loop is the whole demo.
        verbosity_level=2,
    )
