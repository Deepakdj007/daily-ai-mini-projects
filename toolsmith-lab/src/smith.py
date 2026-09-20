"""Turning a write_tool call into a registered tool, or into a reason it is not.

Inputs:  a write_tool call, the task it was made for, and the arm's policy
Outputs: an Outcome carrying the tool, or the exact text sent back to the model

Seven checks, cheapest first, and every one of them free of tokens. The order
matters only in that a rejection should name the first thing wrong rather than
the deepest.

One message in this file is deliberately vague, and it is the most important
one. When a tool passes its own tests and fails the held-out inputs, the model
is told how many failed and nothing else. Naming the inputs would hand it the
test set, and the held-out check would stop measuring whether the code
generalises and start measuring whether the model can patch four more cases.
The rung above it would then be the rung below it with extra examples, and the
whole comparison would be about a mechanism nobody built.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace

from src import config, guard, matcher, registry, sandbox, spec


@dataclass(frozen=True, slots=True)
class Outcome:
    """What happened to one write_tool call."""

    stage: str
    """registered | duplicate | parse | name | collision | hardcode | guard |
    contract | crash | timeout | selftest | holdout"""
    ok: bool
    message: str
    tool: registry.Tool | None = None
    tests_passed: int = 0
    tests_total: int = 0
    holdout_passed: int = 0
    holdout_total: int = 0
    holdout_runnable: bool = True
    guard_layer: str = ""

    @property
    def burns_repair(self) -> bool:
        """A duplicate is a no-op, not a failure, so it does not cost an attempt.

        Under persist a model that forgets to read its tool list will rewrite
        what it already has. Charging that against the repair budget would
        punish the headline arm for something that costs nothing.
        """
        return not self.ok and self.stage != "duplicate"


def _first_disagreement(tests: tuple, results: list[dict]) -> str:
    """The model's own failing test, quoted back. Its own data, so safe to show."""
    for (args, expect), row in zip(tests, results):
        if row.get("ok") and row.get("value") == expect:
            continue
        got = row.get("value") if row.get("ok") else f"raised {row.get('error')}"
        return (f"  args {json.dumps(args, ensure_ascii=False)} -> you expected "
                f"{json.dumps(expect, ensure_ascii=False)}, the code returned "
                f"{json.dumps(got, ensure_ascii=False, default=str)}")
    return ""


def attempt(raw: dict, *, task: dict, family: dict, policy, reg: registry.Registry,
            round_no: int, arm: str) -> Outcome:
    """Check one proposed tool and register it if every enabled check passes."""
    proposal, why = spec.parse(raw)
    if proposal is None:
        return Outcome("parse", False, f"REJECTED (schema): {why}")

    if problem := registry.name_problem(proposal.name):
        return Outcome("name", False, f"REJECTED (name): {problem}")

    clash = reg.collision(proposal.name, proposal.source)
    if clash == "duplicate":
        return Outcome("duplicate", False,
                       f"ALREADY REGISTERED: '{proposal.name}' with identical code is "
                       "in your tool list. Call it.")
    if clash:
        return Outcome("collision", False, f"REJECTED (name): {clash}")

    if problem := spec.hardcode_problem(proposal.source, task["inputs"], task["answer"]):
        return Outcome("hardcode", False, f"REJECTED (hardcoded): {problem}")

    verdict = guard.guard(proposal.source)
    if not verdict.ok:
        return Outcome("guard", False, guard_layer=verdict.layer, message=(
            f"REJECTED (guard, {verdict.layer}): {verdict.reason} Use only the "
            "standard library modules listed in your instructions, and do not try "
            "to work around this."))

    tool = registry.Tool(
        name=proposal.name, description=proposal.description or guard.describe(proposal.source),
        parameters=proposal.schema, source=proposal.source, task_id=task["id"],
        family=task["family"], born_arm=arm, born_round=round_no,
        digest=registry.digest_of(proposal.source))

    if not policy.test:
        # The `write` arm: nothing is checked, and the model is told nothing
        # was. What that costs is the whole point of the rung above.
        reg.add(tool)
        return Outcome("registered", True, tool=tool, message=(
            f"REGISTERED '{tool.name}'. It is in your tool list from the next "
            "message - call it now."))

    held = list(family["holdout"]) if policy.holdout else []
    bound = [spec.bind(proposal.schema, row["inputs"]) for row in held]
    runnable = bool(held) and all(args is not None for args in bound)
    # One sandbox process for the model's own tests and the hidden ones
    # together: twelve checks, one interpreter start.
    calls = [args for args, _ in proposal.tests]
    if runnable:
        calls += [args for args in bound if args is not None]

    results, raw_run = sandbox.execute(proposal.source, calls)
    if raw_run.outcome == "timeout":
        return Outcome("timeout", False, tests_total=len(proposal.tests), message=(
            f"FAILED (timeout): your checks did not finish in "
            f"{config.SANDBOX_TIMEOUT_S:.0f} seconds. Remove any loop that may not "
            "terminate."))
    if raw_run.outcome == "output":
        return Outcome("crash", False, tests_total=len(proposal.tests), message=(
            "FAILED (output): your tool printed more than the runner allows. "
            "Remove the printing and return the answer instead."))

    first = results[0]
    if not first.get("ok") and "name 'run' is not defined" in str(first.get("error", "")):
        return Outcome("contract", False, message=(
            "REJECTED (contract): the module must define run(**kwargs). "
            f"Got: {first.get('error')}"))

    own = results[:len(proposal.tests)]
    passed = sum(row.get("ok") and row.get("value") == expect
                 for row, (_, expect) in zip(own, proposal.tests))
    if passed < len(proposal.tests):
        return Outcome("selftest", False, tests_passed=passed,
                       tests_total=len(proposal.tests), message=(
            f"FAILED (your own tests): {len(proposal.tests) - passed} of "
            f"{len(proposal.tests)} disagreed.\n"
            f"{_first_disagreement(proposal.tests, own)}\n"
            "One of those two is wrong. Work out which by hand before rewriting. "
            "Do not delete or weaken the failing test."))

    if policy.holdout and not runnable:
        # Not a rejection. A tool whose arity does not match the family cannot
        # be checked this way, and refusing it would measure the interface the
        # model chose rather than whether its code is right. Counted, and a
        # gate voids the run if it happens too often to trust the rung.
        reg.add(tool)
        return Outcome("registered", True, tool=tool, holdout_runnable=False,
                       tests_passed=passed, tests_total=len(proposal.tests), message=(
            f"REGISTERED '{tool.name}'. It passed all {passed} of your tests. "
            "It is in your tool list from the next message - call it now."))

    # The held-out inputs are judged with the same matcher that grades the run.
    # A check that disagreed with the grader would reject tools the scorecard
    # would have marked correct, and the rung would be measuring the mismatch.
    hidden = results[len(proposal.tests):]
    held_pass = sum(
        bool(row.get("ok")) and matcher.matches(row.get("value"), expected["expect"],
                                                family["compare"])
        for row, expected in zip(hidden, held))
    if policy.holdout and held_pass < len(held):
        return Outcome("holdout", False, tests_passed=passed,
                       tests_total=len(proposal.tests), holdout_passed=held_pass,
                       holdout_total=len(held), message=(
            f"FAILED (held-out check): your code passes all {passed} of your own "
            f"tests but fails {len(held) - held_pass} of {len(held)} hidden inputs "
            "of the same kind. The hidden inputs are not shown. Your implementation "
            "is probably shaped around the examples you were given - implement the "
            "general rule."))

    tool = replace(tool, tests_passed=passed, holdout_passed=held_pass,
                   holdout_total=len(held))
    reg.add(tool)
    detail = f"passed all {passed} of your tests"
    if policy.holdout:
        detail += f" and {held_pass} held-out checks"
    return Outcome("registered", True, tool=tool, tests_passed=passed,
                   tests_total=len(proposal.tests), holdout_passed=held_pass,
                   holdout_total=len(held), message=(
        f"REGISTERED '{tool.name}' - it {detail}. It is in your tool list from the "
        "next message - call it now."))


def refusal() -> str:
    """What the model is told once its repair attempts are spent."""
    return (f"REFUSED: {config.MAX_REPAIR_ATTEMPTS} attempts at writing a tool have "
            "failed on this task. No more tools will be accepted. Answer with what "
            "you have, or say that you cannot.")
