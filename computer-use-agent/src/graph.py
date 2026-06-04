"""The See -> Think -> Act loop, built as a LangGraph state machine.

Flow: START -> see -> think -> act -> (loop back to see, or END).
The loop ends when Gemini returns the 'done' action or MAX_STEPS is reached.
"""

from typing import Optional, TypedDict

from google import genai
from langgraph.graph import END, START, StateGraph
from PIL import Image

from src.actions import execute_action
from src.brain import Action, decide_action
from src.config import MAX_STEPS
from src.screen import capture_screen


class AgentState(TypedDict):
    """Everything the loop carries from one step to the next."""

    task: str
    screenshot: Optional[Image.Image]
    action: Optional[Action]
    step: int
    done: bool
    history: list[str]


# Created lazily so load_dotenv() has run before the key is read.
_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client()
    return _client


def see(state: AgentState) -> dict:
    """Capture the current screen."""
    return {"screenshot": capture_screen()}


def think(state: AgentState) -> dict:
    """Ask Gemini for the single next action, given what's already been done."""
    action = decide_action(
        _get_client(), state["task"], state["screenshot"], state["history"]
    )
    print(f"[step {state['step'] + 1}] {action.action}: {action.reasoning}")
    return {"action": action}


def act(state: AgentState) -> dict:
    """Execute the action, record a semantic history entry, advance the step."""
    action = state["action"]
    entry = f"{action.action}: {action.reasoning}"
    if action.action == "done":
        return {"done": True, "step": state["step"] + 1,
                "history": state["history"] + [entry]}
    execute_action(action)
    return {"step": state["step"] + 1, "history": state["history"] + [entry]}


def should_continue(state: AgentState) -> str:
    """Loop back to 'see', or stop when done or the step cap is hit."""
    if state["done"] or state["step"] >= MAX_STEPS:
        return "end"
    return "see"


def build_graph():
    """Assemble and compile the See -> Think -> Act graph."""
    g = StateGraph(AgentState)
    g.add_node("see", see)
    g.add_node("think", think)
    g.add_node("act", act)
    g.add_edge(START, "see")
    g.add_edge("see", "think")
    g.add_edge("think", "act")
    g.add_conditional_edges("act", should_continue, {"see": "see", "end": END})
    return g.compile()
