"""Assembles the crew and runs one analysis.

Inputs:  a ticker symbol and whether to run the analysts in parallel.
Outputs: build_crew() for the assembled Crew, run_analysis() for a finished
         note plus the wall-clock time it took.
"""

import time
from dataclasses import dataclass

from crewai import Crew, Process

from src.agents import build_agents
from src.config import MAX_RPM
from src.tasks import build_tasks


@dataclass
class Result:
    """One completed analysis."""

    ticker: str
    note: str
    seconds: float
    parallel: bool


def build_crew(parallel: bool = True) -> Crew:
    """Assemble the five agents and five tasks into a crew.

    Three settings here are not defaults and all three matter:

    process=sequential — async_execution is a sequential-process feature. The
    analysts still run concurrently; "sequential" describes how CrewAI walks the
    task list, not whether it waits.

    memory=False — Crew's default embedder is OpenAI, so switching memory on
    demands an OPENAI_API_KEY even though every agent here is on Gemini. This
    project has no use for memory across runs.

    max_rpm — four analysts fire at once, so it is the burst that trips free-tier
    rate limits rather than the total number of calls.
    """
    agents = build_agents()
    tasks = build_tasks(agents, parallel=parallel)
    return Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential,
        memory=False,
        max_rpm=MAX_RPM,
        # Without this, a first-time run blocks on an interactive tracing prompt.
        tracing=False,
        verbose=True,
    )


def run_analysis(ticker: str, parallel: bool = True) -> Result:
    """Run the crew end to end for one ticker and time it.

    The ticker reaches every task through CrewAI's input interpolation: the task
    descriptions contain a ticker placeholder in braces, and kickoff fills it.

    Args:
        ticker: an exact symbol, e.g. "RELIANCE.NS".
        parallel: False forces the analysts to run one at a time.

    Returns:
        A Result carrying the finished note and the elapsed seconds.
    """
    crew = build_crew(parallel=parallel)
    started = time.perf_counter()
    output = crew.kickoff(inputs={"ticker": ticker})
    elapsed = time.perf_counter() - started
    return Result(
        ticker=ticker,
        note=str(output).strip(),
        seconds=elapsed,
        parallel=parallel,
    )
