"""The five agents: four specialist analysts and one editor.

Each analyst carries exactly one tool. The editor carries none — it works purely
from the four analyst outputs that CrewAI hands it as context.

Inputs:  config for model ids and rate limits, tools for the data access.
Outputs: build_agents() returning the five configured Agents.

Every agent sets llm= explicitly. CrewAI defaults to OpenAI for any agent that
does not, which surfaces as a confusing OPENAI_API_KEY error halfway through a
run that is supposed to be entirely on Gemini.
"""

from crewai import Agent

from src import tools
from src.config import MAX_ITER, MAX_RETRIES, MAX_RPM, MODEL_ANALYST, MODEL_EDITOR, make_llm


def _analyst(role: str, goal: str, backstory: str, tool: object) -> Agent:
    """Build one specialist analyst with a single tool.

    inject_date puts today's date in the prompt. A stock note that does not know
    what day it is will happily describe a stale quarter as current.
    """
    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        tools=[tool],
        llm=make_llm(MODEL_ANALYST, temperature=0.2),
        max_iter=MAX_ITER,
        max_rpm=MAX_RPM,
        max_retry_limit=MAX_RETRIES,
        inject_date=True,
        allow_delegation=False,
        verbose=True,
    )


def build_agents() -> dict[str, Agent]:
    """Create all five agents, keyed by short name."""
    price = _analyst(
        role="Technical Analyst",
        goal=(
            "Describe how the share price has actually behaved and what the trend, "
            "moving averages and volatility say about momentum and risk."
        ),
        backstory=(
            "You read price action for a living and you distrust narratives that the "
            "chart does not support. You always say where price sits relative to its "
            "50-day and 200-day averages and its 52-week range, and you call a "
            "downtrend a downtrend."
        ),
        tool=tools.price_snapshot,
    )

    fundamentals = _analyst(
        role="Fundamental Analyst",
        goal=(
            "Judge business quality and valuation from the filed statements: growth, "
            "margins, returns, cash generation, leverage and what multiple the market "
            "is paying."
        ),
        backstory=(
            "You came up auditing balance sheets, so you trust cash flow over "
            "headline profit and you always check debt before getting excited about "
            "growth. You state plainly whether the current multiple looks demanding "
            "or cheap against the growth on offer."
        ),
        tool=tools.fundamentals_snapshot,
    )

    news = _analyst(
        role="News and Catalyst Analyst",
        goal=(
            "Identify the current narrative around the company and separate genuine "
            "catalysts and risks from routine market noise."
        ),
        backstory=(
            "You have watched too many investors buy a headline and sell a fact. You "
            "group coverage into themes, flag which items could actually move earnings, "
            "and ignore stories that merely mention the company in passing."
        ),
        tool=tools.news_snapshot,
    )

    street = _analyst(
        role="Sell-side Consensus Analyst",
        goal=(
            "Report what professional analysts currently expect, how that has shifted, "
            "and how much of the expected upside is already priced in."
        ),
        backstory=(
            "You track the sell side without deferring to it. You note when the "
            "ratings spread has drifted, compare the mean target with the live price, "
            "and treat a crowded bullish consensus as a risk rather than a reassurance."
        ),
        tool=tools.street_snapshot,
    )

    editor = Agent(
        role="Head of Equity Research",
        goal=(
            "Turn four independent analyst briefings into one coherent research note "
            "that a reader can act on, including where the analysts disagree."
        ),
        backstory=(
            "You have signed off on thousands of research notes. You write in plain "
            "English, lead with the conclusion, and refuse to paper over a "
            "contradiction between two analysts — where they disagree, you say so and "
            "explain which evidence you weight more heavily."
        ),
        llm=make_llm(MODEL_EDITOR, temperature=0.4),
        max_iter=MAX_ITER,
        max_rpm=MAX_RPM,
        max_retry_limit=MAX_RETRIES,
        inject_date=True,
        allow_delegation=False,
        verbose=True,
    )

    return {
        "price": price,
        "fundamentals": fundamentals,
        "news": news,
        "street": street,
        "editor": editor,
    }
