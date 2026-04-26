# state.py
import operator
from typing import Annotated, Optional
from typing_extensions import TypedDict


class AgentSignal(TypedDict):
    """Output from a single specialist agent."""
    agent: str           # "price", "sentiment", "onchain", "macro", "risk"
    signal: str          # "BULLISH", "BEARISH", or "NEUTRAL"
    confidence: float    # 0.0 to 1.0
    summary: str         # one-line explanation
    raw_data: dict       # the numbers behind the call


class MarketMindState(TypedDict):
    """Shared state that flows through the entire graph."""
    ticker: str
    asset_type: str      # "stock" or "crypto"

    # operator.add is the reducer — it APPENDS each agent's result
    # instead of overwriting. This prevents race conditions when all
    # 5 agents try to write to the same field simultaneously.
    agent_signals: Annotated[list, operator.add]

    # Synthesis agent writes these after all 5 agents complete
    final_verdict: Optional[str]
    final_confidence: Optional[float]
    final_reasoning: Optional[str]