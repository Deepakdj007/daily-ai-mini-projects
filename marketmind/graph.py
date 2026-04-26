# graph.py
from langgraph.graph import StateGraph, START, END
from state import MarketMindState
from agents.price_agent import price_agent
from agents.sentiment_agent import sentiment_agent
from agents.onchain_agent import onchain_agent
from agents.macro_agent import macro_agent
from agents.risk_agent import risk_agent
from agents.synthesis_agent import synthesis_agent


def validate_input(state: MarketMindState) -> dict:
    """Normalize ticker to uppercase and detect asset type."""
    ticker = state["ticker"].strip().upper()
    crypto_tickers = {
        "BTC", "ETH", "SOL", "BNB", "DOGE", "ADA",
        "AVAX", "DOT", "MATIC", "LINK", "XRP", "LTC",
    }
    return {
        "ticker": ticker,
        "asset_type": "crypto" if ticker in crypto_tickers else "stock",
        "agent_signals": [],   # reset before each run
    }


def build_graph():
    graph = StateGraph(MarketMindState)

    # Register all nodes
    graph.add_node("validate", validate_input)
    graph.add_node("price_agent", price_agent)
    graph.add_node("sentiment_agent", sentiment_agent)
    graph.add_node("onchain_agent", onchain_agent)
    graph.add_node("macro_agent", macro_agent)
    graph.add_node("risk_agent", risk_agent)
    graph.add_node("synthesis_agent", synthesis_agent)

    # Entry point
    graph.add_edge(START, "validate")

    # FAN-OUT: validate fires all 5 agents simultaneously.
    # LangGraph detects multiple edges leaving one node and runs
    # all destinations as a single superstep — all at once.
    graph.add_edge("validate", "price_agent")
    graph.add_edge("validate", "sentiment_agent")
    graph.add_edge("validate", "onchain_agent")
    graph.add_edge("validate", "macro_agent")
    graph.add_edge("validate", "risk_agent")

    # FAN-IN: synthesis waits for all 5 agents to complete first.
    # LangGraph's sync barrier enforces this automatically.
    graph.add_edge("price_agent", "synthesis_agent")
    graph.add_edge("sentiment_agent", "synthesis_agent")
    graph.add_edge("onchain_agent", "synthesis_agent")
    graph.add_edge("macro_agent", "synthesis_agent")
    graph.add_edge("risk_agent", "synthesis_agent")

    graph.add_edge("synthesis_agent", END)

    return graph.compile()


# Compile once at module load — reuse for all requests
marketmind_graph = build_graph()