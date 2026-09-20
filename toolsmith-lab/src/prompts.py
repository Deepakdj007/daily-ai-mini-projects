"""What the agent is told, and how that changes between arms.

Inputs:  an arm's policy
Outputs: the system prompt for that arm

The rules about writing a tool are adapted from the self-healing agent earlier
in this series, which had to tell a model not to delete the failing assertion.
Two of them are new here and both exist because of a specific way to cheat.

"Implement the general rule" is the answer to a tool shaped around the one
input it was shown. The prompt alone does not stop that - a prompt is not a
control - which is exactly why the held-out inputs exist and why the rung that
runs them is in the ladder. The instruction is here so that the arms which do
not run hidden inputs are not being sabotaged by a missing sentence: every arm
is told to generalise, and only some arms are made to.

"Call it, do not answer from memory" is the answer to an agent that writes a
perfectly good tool and then ignores it. That one is checked rather than
trusted: the run records which tools were actually called.
"""

from __future__ import annotations

BASE = (
    "You answer one question at a time, exactly.\n\n"
    "Before answering, decide whether the answer follows from a rule applied "
    "step by step - looping over the characters or items of an input, looking "
    "something up in a table, building a string piece by piece, or walking a "
    "schedule row by row. If it does, the answer must come from a tool that "
    "implements the whole rule and returns the finished answer. The calculator "
    "evaluates one arithmetic expression and nothing more; it cannot carry a "
    "rule, and stringing calculator steps together by hand is the same as "
    "doing it in your head. Both produce answers that look right and are not."
    "\n\n"
    "Give the answer in the form the task asks for and nothing else - no "
    "working, no units that were not asked for, no explanation. Call "
    "submit_answer exactly once, at the end."
)

WRITE_RULES = (
    "\n\nIf no tool in your list implements the rule, write one with "
    "write_tool, then call it.\n\n"
    "Rules for a tool you write:\n"
    "- One module. It must define run(**kwargs) and return a JSON-serialisable "
    "value.\n"
    "- Standard library only, from: {allowed}. No network, no filesystem, no "
    "subprocess, no eval, no getattr.\n"
    "- Module level may hold only imports, defs and literal constants. Write a "
    "lookup table out as a nested tuple rather than building it with a loop.\n"
    "- It runs in a fresh process on every call, so it cannot keep state "
    "between calls.\n"
    "- Implement the GENERAL rule, not the example in front of you. Your tool "
    "will be run on inputs you have not seen. A tool with this task's values "
    "written into it will be rejected.\n"
    "- Your tests must say what the answer SHOULD be, worked out by hand from "
    "the rule. Do not run the code in your head and write down whatever it "
    "would produce - that tests nothing. If a test fails, one of the two is "
    "wrong; work out which before you rewrite. Do not delete or weaken a "
    "failing test to make it pass.\n"
    "- Once a tool is registered it appears in your tool list. Call it. Do not "
    "answer from memory, and do not write a second tool that does the same job."
)

INLINE_RULES = (
    "\n\nWhen a question needs a computation, write a snippet and run it with "
    "run_python. It must define run(**kwargs); it is called with no arguments, "
    "so put the values you need inside it. Standard library only, from: "
    "{allowed}. Nothing is kept between tasks - if you need the same "
    "calculation again later you will write it again."
)

NO_TOOLS = (
    "\n\nYou have no tools. Answer from what you know."
)


def system_prompt(policy) -> str:
    """The system message for one arm."""
    from src import config

    allowed = ", ".join(sorted(config.ALLOWED_IMPORTS))
    text = BASE
    if policy.write:
        text += WRITE_RULES.format(allowed=allowed)
    elif policy.inline:
        text += INLINE_RULES.format(allowed=allowed)
    elif not policy.tools:
        text += NO_TOOLS
    return text


if __name__ == "__main__":
    from src import ladder

    for arm in ladder.ALL_ARMS:
        text = system_prompt(arm.policy)
        print(f"{arm.name:<9} {len(text):>4} chars")
        if arm.policy.write:
            assert "GENERAL rule" in text and "weaken a failing test" in text
        if arm.policy.inline:
            assert "run_python" in text and "write_tool" not in text
        if not arm.policy.tools:
            assert "no tools" in text
    # Every arm that writes code is told the same thing about generalising, so
    # the rungs differ in what is enforced rather than in what was asked.
    writers = [system_prompt(a.policy) for a in ladder.ALL_ARMS if a.policy.write]
    assert len(set(writers)) == 1, "the write arms must share one prompt"
    print("\nevery write arm shares one prompt, so the rungs differ in enforcement")
    print("OK - prompts")
