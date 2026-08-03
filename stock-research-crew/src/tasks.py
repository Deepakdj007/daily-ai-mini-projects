"""The five tasks, and the one flag that makes this project parallel.

The four analyst tasks carry async_execution=True, so CrewAI dispatches them
together instead of waiting for each to finish. The editor task is synchronous
and lists all four in context=, which is what makes it wait for them.

That ordering is not a style choice. CrewAI validates it: a crew may end with at
most one asynchronous task, so if the editor were async too the whole run fails
with a ValidationError before a single token is spent.

Inputs:  the agents dict from src/agents.py, and whether to run in parallel.
Outputs: build_tasks() returning the five Tasks in execution order.

The literal "ticker" placeholder in braces is filled by CrewAI at kickoff from
inputs. Avoid any other braces in these strings — CrewAI treats them all as
interpolation slots.
"""

from crewai import Agent, Task

ANALYST_OUTPUT = (
    "A briefing of 120-200 words in plain prose, no preamble. Open with your single "
    "most important finding, support it with the specific numbers you retrieved, and "
    "close with the clearest risk you can see from your angle. Quote figures exactly "
    "as the tool reported them, including the currency unit. If the tool reported data "
    "as unavailable, say so plainly instead of guessing."
)


def build_tasks(agents: dict[str, Agent], parallel: bool = True) -> list[Task]:
    """Create the five tasks in execution order.

    Args:
        agents: the dict returned by build_agents().
        parallel: True runs the four analysts concurrently. False forces them to
            run one after another, which is only useful for the timing comparison.

    Returns:
        The four analyst tasks followed by the editor task.
    """
    price = Task(
        description=(
            "Analyse the recent price behaviour of {ticker}. Call your tool once with "
            "the ticker exactly as written. Then report the trend, where price sits "
            "against its 50-day and 200-day moving averages and its 52-week range, "
            "how volatile it has been, and what that combination implies about "
            "momentum right now."
        ),
        expected_output=ANALYST_OUTPUT,
        agent=agents["price"],
        async_execution=parallel,
    )

    fundamentals = Task(
        description=(
            "Assess the business and valuation of {ticker}. Call your tool once with "
            "the ticker exactly as written. Then report revenue and profit growth, "
            "margins, return on equity, cash generation against reported profit, the "
            "debt position, and whether the current multiples look demanding or cheap "
            "for the growth on offer."
        ),
        expected_output=ANALYST_OUTPUT,
        agent=agents["fundamentals"],
        async_execution=parallel,
    )

    news = Task(
        description=(
            "Establish the current narrative around {ticker}. Call your tool once with "
            "the ticker exactly as written. Then group the coverage into themes, name "
            "the specific catalysts that could move earnings, and name the risks. "
            "Ignore stories that only mention the company in passing, and say so if "
            "the coverage is thin."
        ),
        expected_output=ANALYST_OUTPUT,
        agent=agents["news"],
        async_execution=parallel,
    )

    street = Task(
        description=(
            "Report the sell-side view on {ticker}. Call your tool once with the ticker "
            "exactly as written. Then give the current ratings spread, how it has "
            "shifted over the past two months, the mean price target against the live "
            "price, the implied upside, and the next earnings date. Say whether the "
            "consensus looks crowded."
        ),
        expected_output=ANALYST_OUTPUT,
        agent=agents["street"],
        async_execution=parallel,
    )

    editor = Task(
        description=(
            "Write the equity research note on {ticker} using the four analyst "
            "briefings you have been given as context. Do not invent any figure that "
            "does not appear in them.\n\n"
            "Structure the note with these markdown sections, in this order:\n"
            "## Verdict — three or four sentences giving your overall read and the "
            "single most important reason for it.\n"
            "## The Bull Case — the strongest honest argument for owning it.\n"
            "## The Bear Case — the strongest honest argument against.\n"
            "## Where the Analysts Disagree — name the actual tension between the four "
            "briefings and say which evidence you weight more heavily. If they broadly "
            "agree, say that and explain what would change the picture.\n"
            "## What to Watch — three specific, checkable things, each on its own "
            "bullet, with the date or level that would make it matter.\n\n"
            "Write for an intelligent reader who is not a professional investor. Plain "
            "English, short paragraphs, no filler, no restating the section titles. Do "
            "not add a top-level heading or a disclaimer — both are added around your "
            "output."
        ),
        expected_output=(
            "A markdown research note using exactly the five level-two headings "
            "specified, 500-750 words in total, containing only figures that appear in "
            "the analyst briefings."
        ),
        agent=agents["editor"],
        context=[price, fundamentals, news, street],
        # Never async: the crew must end with at most one asynchronous task, and
        # this one is also the join point for the four above.
        async_execution=False,
        markdown=True,
    )

    return [price, fundamentals, news, street, editor]
