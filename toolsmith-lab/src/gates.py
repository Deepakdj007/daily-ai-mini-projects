"""Every check that costs nothing, run before anything is spent.

    PYTHONPATH=. uv run python -m src.gates

Inputs:  data/tasks.json, the oracle, and the shortcut implementations
Outputs: a list of problems; exits non-zero if there are any

A task that can be answered some other way measures nothing, and it fails
silently: the arm scores well, the ladder looks monotone, and the leaderboard
is wrong in a way nobody has a reason to check. These run before the run, which
is the only time catching it is cheap.

Two of them are the ones the whole ladder rests on. `the shortcut is wrong`
says the families are hard, so an arm that scores well did something. `the
held-out inputs discriminate` says there exists an implementation that passes
all four visible tasks and fails the hidden ones - which is the only reason the
holdout rung is in the ladder at all. If that gate ever passes vacuously, the
rung is decoration and the report should say so rather than claim a mechanism.
"""

from __future__ import annotations

import itertools
import operator
import re

from src import attacks, corpus, guard, matcher, oracle, shortcuts, tools

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_OPS = (operator.add, operator.sub, operator.mul, operator.truediv,
        operator.floordiv, operator.mod)


def _numbers(text: str) -> list[float]:
    """Numeric literals in a prompt, commas removed."""
    found = []
    for raw in _NUMBER.findall(text):
        try:
            found.append(float(raw.replace(",", "")))
        except ValueError:
            continue
    return found[:12]


def _reachable(text: str) -> set[float]:
    """Everything one calculator call could produce from this prompt.

    The literals themselves, every pair combined by an arithmetic operator, and
    what the other two fixed tools return. If a family's answer is in here, the
    fixed baseline can reach it and the family is not measuring tool-writing.
    """
    values = _numbers(text)
    reachable = set(values)
    for left, right in itertools.permutations(values, 2):
        for op in _OPS:
            try:
                result = op(left, right)
            except (ZeroDivisionError, OverflowError, ValueError):
                continue
            if isinstance(result, (int, float)) and abs(result) < 1e15:
                reachable.add(float(result))
    reachable.update(_numbers(tools.text_stats(text)))
    return reachable


def distinct_answers() -> list[str]:
    """A family whose four answers carry no information measures nothing."""
    problems = []
    for entry in corpus.families():
        seen = {str(matcher.normalise(task["answer"], entry["compare"]))
                for task in entry["tasks"]}
        if len(seen) < 3:
            problems.append(f"{entry['family']}: only {len(seen)} distinct answers "
                            "across four tasks")
    return problems


def shortcut_is_wrong() -> list[str]:
    """Someone who does no computation must get nearly all of them wrong."""
    problems = []
    for entry in corpus.families():
        right = 0
        for task in entry["tasks"]:
            try:
                guess = shortcuts.TRIVIAL[entry["family"]](*task["inputs"])
            except Exception:  # noqa: BLE001 - a shortcut that crashes is wrong
                continue
            right += matcher.matches(guess, task["answer"], entry["compare"])
        if right > 1:
            problems.append(f"{entry['family']}: the trivial answer is right on "
                            f"{right} of 4 tasks - the family is too easy")
    return problems


def fixed_cannot_reach() -> list[str]:
    """No large numeric answer may be one calculator call away from its prompt.

    Only the money family qualifies. For an answer drawn from ten possible
    digits, "reachable by some arithmetic expression over the prompt" is true
    of every digit, so the check would fire on a perfectly good family and tell
    you nothing - it measures the size of the answer space, not the difficulty
    of the task. A small answer space is a real weakness, but the gate that
    speaks to it is the null arm's score, not this one.
    """
    problems = []
    for entry in corpus.families():
        if entry["compare"] != "money2":
            continue
        for task in entry["tasks"]:
            target = matcher.normalise(task["answer"], entry["compare"])
            if any(matcher.matches(value, target, entry["compare"])
                   for value in _reachable(task["prompt"])):
                problems.append(f"{task['id']}: the answer is one arithmetic step "
                                "from the numbers in its own prompt")
    return problems


def holdout_is_hidden() -> list[str]:
    """No held-out input may appear in any prompt the model is shown.

    That includes the worked example in the contract. An example that happened
    to be a held-out input would hand the model the answer to a hidden check,
    and the rung that runs those checks would stop measuring anything.
    """
    problems = []
    for entry in corpus.families():
        prompts = " ".join(task["prompt"] for task in entry["tasks"])
        for row in entry["holdout"]:
            for value in row["inputs"]:
                for piece in (value if isinstance(value, list) else [value]):
                    # Strings only. A held-out number shares digits with a
                    # visible prompt all the time - the 0.0 of an interest-free
                    # loan is inside the "10.0%" of another task - and what
                    # matters for a numeric family is the whole argument tuple,
                    # which tools/author_tasks.py already refuses to duplicate.
                    if isinstance(piece, str) and len(piece) >= 3 and piece in prompts:
                        problems.append(f"{entry['family']}: held-out input "
                                        f"{piece!r} appears in a visible prompt")
    return problems


def example_is_separate() -> list[str]:
    """The worked example must be neither a graded task nor a hidden check."""
    problems = []
    for entry in corpus.families():
        example = str(entry.get("example"))
        for task in entry["tasks"]:
            if str(task["inputs"]) == example:
                problems.append(f"{entry['family']}: the worked example is task "
                                f"{task['id']}, so the answer is given away")
        for row in entry["holdout"]:
            if str(row["inputs"]) == example:
                problems.append(f"{entry['family']}: the worked example is one of "
                                "the held-out inputs")
    return problems


def holdout_discriminates() -> list[str]:
    """There must exist code that passes all four tasks and fails the hidden ones.

    Without this the holdout rung cannot do anything, and a rung that cannot do
    anything must not be reported as a mechanism.
    """
    problems = []
    for entry in corpus.families():
        overfit = shortcuts.OVERFIT[entry["family"]]
        visible = 0
        for task in entry["tasks"]:
            try:
                visible += matcher.matches(overfit(*task["inputs"]), task["answer"],
                                           entry["compare"])
            except Exception:  # noqa: BLE001
                pass
        missed = 0
        for row in entry["holdout"]:
            try:
                missed += not matcher.matches(overfit(*row["inputs"]), row["expect"],
                                              entry["compare"])
            except Exception:  # noqa: BLE001 - a crash is a failure
                missed += 1
        if visible < len(entry["tasks"]):
            problems.append(f"{entry['family']}: the overfit implementation fails "
                            f"{len(entry['tasks']) - visible} visible task(s), so it "
                            "is not the kind of mistake a self-test would miss")
        if missed == 0:
            problems.append(f"{entry['family']}: the overfit implementation passes "
                            "every held-out input, so the holdout rung has nothing "
                            "to catch here")
    return problems


def answers_are_the_oracle() -> list[str]:
    """Every gold answer recomputed, through the matcher that grades the run."""
    problems = []
    for entry in corpus.families():
        for task in entry["tasks"]:
            fresh = oracle.solve_positional(entry["family"], task["inputs"])
            if not matcher.matches(fresh, task["answer"], entry["compare"]):
                problems.append(f"{task['id']}: stored {task['answer']!r}, the oracle "
                                f"now says {fresh!r}")
        for index, row in enumerate(entry["holdout"]):
            fresh = oracle.solve_positional(entry["family"], row["inputs"])
            if not matcher.matches(fresh, row["expect"], entry["compare"]):
                problems.append(f"{entry['family']} holdout {index}: stored "
                                f"{row['expect']!r}, the oracle says {fresh!r}")
    return problems


def contract_is_uniform() -> list[str]:
    """A family's contract must be byte-identical in all four of its prompts.

    Otherwise a tool written for the first task cannot be reused on the second,
    and the reuse figure measures prompt drift instead of a registry.
    """
    return [f"{entry['family']}: the contract differs between prompts"
            for entry in corpus.families()
            if any(entry["contract"] not in task["prompt"] for task in entry["tasks"])]


def corpus_is_stamped() -> list[str]:
    """The corpus must have been built by the oracle that is installed now."""
    if corpus.load().get("oracle") != oracle.digest():
        return ["data/tasks.json was built by a different oracle. Re-run: "
                "PYTHONPATH=. uv run python -m tools.author_tasks"]
    return []


def guard_admits_honest_tools() -> list[str]:
    """The guard must pass ordinary tools, or every arm loses to its own guard."""
    return [f"the guard blocks an honest tool ({name}): {guard.guard(source).reason[:60]}"
            for name, source, _, _ in attacks.BENIGN if not guard.guard(source).ok]


CHECKS = (
    ("answers are the oracle's", answers_are_the_oracle),
    ("the corpus matches the oracle", corpus_is_stamped),
    ("four distinct answers per family", distinct_answers),
    ("the trivial shortcut is wrong", shortcut_is_wrong),
    ("the fixed toolset cannot reach the answer", fixed_cannot_reach),
    ("held-out inputs are never shown", holdout_is_hidden),
    ("the worked example is neither graded nor hidden", example_is_separate),
    ("held-out inputs discriminate", holdout_discriminates),
    ("the contract is uniform within a family", contract_is_uniform),
    ("the guard admits honest tools", guard_admits_honest_tools),
)


def run_all() -> list[tuple[str, list[str]]]:
    """Every gate, with whatever it found."""
    return [(name, check()) for name, check in CHECKS]


if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="validity gates - these run before a token is spent")
    table.add_column("gate")
    table.add_column("")
    table.add_column("detail", overflow="fold")

    failed = 0
    for name, problems in run_all():
        failed += len(problems)
        table.add_row(name, "[green]pass[/]" if not problems else "[red]FAIL[/]",
                      "\n".join(problems[:3]) or "-")
    console.print(table)
    console.print(f"\n[bold]{'all gates pass' if not failed else f'{failed} problem(s)'}[/]")
    raise SystemExit(1 if failed else 0)
