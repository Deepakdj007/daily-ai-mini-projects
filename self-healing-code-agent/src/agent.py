"""The self-healing code agent, built on Pydantic AI + Groq.

The agent receives a coding task, writes a Python script that contains both the
solution and `assert`-based tests, and runs it through the `run_python` tool. If
the script fails, the tool hands the traceback straight back to the model, which
rewrites the code and tries again. The loop ends when every assertion passes.

Inputs:  a natural-language coding task (string).
Outputs: a FinalCode (the working source + a one-line summary) plus a recorded
         log of every attempt for the CLI to replay.
"""

import os
from dataclasses import dataclass, field

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider

from src.config import MODEL
from src.executor import run_code


class FinalCode(BaseModel):
    """The agent's final answer once its own tests pass."""

    code: str       # the complete, working Python script
    summary: str    # one sentence on what it does and how it was fixed


@dataclass
class Attempt:
    """One write-and-run cycle, kept so the CLI can show the heal loop."""

    code: str
    ok: bool
    output: str     # stdout on success, traceback on failure


@dataclass
class Deps:
    """Per-run state shared with the tool via RunContext."""

    attempts: list[Attempt] = field(default_factory=list)


SYSTEM_PROMPT = (
    "You are a Python engineer that writes correct, tested code.\n"
    "For the user's task, write ONE self-contained Python script that:\n"
    "1. Implements the requested function(s).\n"
    "2. Includes several `assert` statements that test it against clear, "
    "correct expected values, covering edge cases.\n"
    "3. Prints 'ALL TESTS PASSED' on the last line.\n"
    "Then call the `run_python` tool with that script to verify it.\n"
    "If the tool reports a failure, read the traceback carefully, fix the bug, "
    "and call `run_python` again with the corrected script. Repeat until it "
    "passes. Do NOT remove or weaken assertions to make tests pass — fix the "
    "actual code.\n"
    "Only after the tool confirms success, return the final working script and a "
    "one-sentence summary."
)


def build_agent() -> Agent[Deps, FinalCode]:
    """Construct the Groq-backed agent with its self-healing tool registered."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
            "from https://console.groq.com/keys"
        )

    model = GroqModel(MODEL, provider=GroqProvider(api_key=api_key))
    agent = Agent(
        model,
        deps_type=Deps,
        output_type=FinalCode,
        system_prompt=SYSTEM_PROMPT,
    )

    @agent.tool
    def run_python(ctx: RunContext[Deps], code: str) -> str:
        """Run a Python script and return whether its tests passed.

        On failure the traceback is returned verbatim so the model can fix it.
        """
        result = run_code(code)
        ctx.deps.attempts.append(
            Attempt(
                code=code,
                ok=result.ok,
                output=result.stdout if result.ok else result.stderr,
            )
        )
        if result.ok:
            return f"SUCCESS. Output:\n{result.stdout}"
        return (
            f"FAILED. The script raised an error:\n{result.stderr}\n"
            "Fix the code and call run_python again."
        )

    return agent
