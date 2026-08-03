"""The four tools, one per analyst.

Each tool is a thin wrapper over src/market.py that returns a compact text
block. Scoping one tool to one agent is deliberate: an agent that can only
fetch its own slice keeps a small context and cannot wander into someone
else's job.

Inputs:  a ticker symbol string, supplied by the agent from its task description.
Outputs: CrewAI BaseTool instances imported by src/agents.py.
"""

from crewai.tools import tool

from src import market


@tool("Price and technicals")
def price_snapshot(ticker: str) -> str:
    """Fetch price history for a stock ticker and return trend, moving averages,
    52-week position, annualised volatility and average volume. Use this for any
    question about how the share price has behaved. Pass the exact ticker,
    including its exchange suffix, e.g. RELIANCE.NS."""
    return market.price_block(ticker)


@tool("Company fundamentals")
def fundamentals_snapshot(ticker: str) -> str:
    """Fetch filed financial statements for a stock ticker and return revenue and
    profit trend, margins, return on equity, cash flow, debt and valuation
    multiples. Use this for any question about business or financial quality.
    Pass the exact ticker, including its exchange suffix, e.g. RELIANCE.NS."""
    return market.fundamentals_block(ticker)


@tool("Recent news")
def news_snapshot(ticker: str) -> str:
    """Fetch recent news headlines and summaries for a stock ticker. Use this to
    find catalysts, risks and the current narrative around the company. Pass the
    exact ticker, including its exchange suffix, e.g. RELIANCE.NS."""
    return market.news_block(ticker)


@tool("Analyst consensus")
def street_snapshot(ticker: str) -> str:
    """Fetch sell-side analyst data for a stock ticker: the buy/hold/sell spread
    now and two months ago, consensus label, mean and range of price targets,
    implied upside, and the next earnings date. Pass the exact ticker, including
    its exchange suffix, e.g. RELIANCE.NS."""
    return market.street_block(ticker)
