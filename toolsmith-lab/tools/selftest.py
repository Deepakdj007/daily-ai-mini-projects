"""Every check that costs no API tokens. Run it before spending any.

    PYTHONPATH=. uv run python -m tools.selftest

Inputs:  nothing
Outputs: one line per check, and a non-zero exit code if any failed

Exits non-zero, so it works as a pre-commit gate.

Every file this touches is redirected into a temporary directory before
anything that writes one is imported, so running the self-test can never
disturb a real run's cache, ledger, results or written tools.

The last suite is the one worth reading: it replaces the single expensive call
in the whole project with a stub and runs the harness end to end. That proves
the wiring - resume, arm order, the results file, the per-arm registry - without
a single token, and wiring is where a long run actually breaks.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from src import config

# Redirect every file this writes before importing anything that writes one.
_TEMP = Path(tempfile.mkdtemp(prefix="toolsmith-selftest-"))
config.OUTPUT_DIR = _TEMP
config.GENERATED_DIR = _TEMP / "generated"
config.CACHE_PATH = _TEMP / "cache.sqlite"
config.LEDGER_PATH = _TEMP / "ledger.sqlite"
config.LEADERBOARD_PATH = _TEMP / "leaderboard.md"
config.results_path = lambda model="": _TEMP / f"results-{(model or 'x').replace('/', '-')}.json"
config.ensure_dirs()

from src import (agent, gates, guard, harness, ladder, matcher, protocol,  # noqa: E402
                 registry, report, sandbox, spec)  # noqa: E402

FAILURES: list[str] = []


def check(name: str, got, want) -> None:
    """Assert and record, rather than stopping at the first failure."""
    if got == want:
        print(f"  [PASS] {name}")
        return
    FAILURES.append(name)
    print(f"  [FAIL] {name}\n         got  {got!r}\n         want {want!r}")


def test_guard() -> None:
    """The static layer blocks what it should and admits what it should."""
    print("guard")
    check("os is not importable",
          guard.guard("import os\ndef run(**kwargs):\n    return 1\n").layer, "imports")
    check("dunders are refused",
          guard.guard("def run(**kwargs):\n    return ().__class__\n").layer, "names")
    check("getattr is refused",
          guard.guard("def run(**kwargs):\n    return getattr(str, 'x')\n").layer, "calls")
    check("a top-level call is refused",
          guard.guard("print(1)\ndef run(**kwargs):\n    return 1\n").layer, "statements")
    check("the sentinel cannot be forged",
          guard.guard(f"def run(**kwargs):\n    return '{protocol.SENTINEL}'\n").layer,
          "sentinel")
    check("an honest tool passes",
          guard.guard("import math\ndef run(**kwargs):\n"
                      "    return math.floor(kwargs['n'])\n").ok, True)


def test_sandbox() -> None:
    """The runtime layer contains what the static layer cannot see."""
    print("sandbox")
    ok, _ = sandbox.execute("def run(**kwargs):\n    return kwargs['n'] * 2\n",
                            [{"n": 3}, {"n": 4}])
    check("a batch runs in one process", [r["value"] for r in ok], [6, 8])
    leak, _ = sandbox.execute(
        "import os\ndef run(**kwargs):\n    return os.environ.get('GROQ_API_KEY')\n", [{}])
    check("the api key is not in the child", leak[0]["value"], None)
    wrote, _ = sandbox.execute(
        "def run(**kwargs):\n    open('x.txt', 'w').write('x')\n    return 1\n", [{}])
    check("writing a file is denied", wrote[0]["ok"], False)
    check("an infinite loop is killed",
          sandbox.run("def run(**kwargs):\n    while True:\n        pass\n", [{}]).outcome,
          "timeout")


def test_protocol() -> None:
    """Result lines survive whatever a tool prints around them."""
    print("protocol")
    parsed, stray = protocol.parse_lines(
        f'noise\n{protocol.SENTINEL}{{"ok": true, "value": 7}}\r\n', 1)
    check("a result line is read past the tool's own output", parsed[0]["value"], 7)
    check("the tool's output is kept separately", stray, "noise")
    short, _ = protocol.parse_lines(f'{protocol.SENTINEL}{{"ok": true, "value": 1}}\n', 3)
    check("a short batch reports the missing calls", len(short), 3)


def test_spec() -> None:
    """A proposal is parsed, and a hard-coded answer is caught."""
    print("spec")
    check("line arrays become a module",
          spec.join_source(["def run(**kwargs):", "    return 1"]),
          "def run(**kwargs):\n    return 1\n")
    check("a schema without additionalProperties is refused",
          bool(spec.schema_problem({"type": "object", "required": ["n"],
                                    "properties": {"n": {}}})), True)
    schema = {"type": "object", "additionalProperties": False,
              "required": ["gstin"], "properties": {"gstin": {"type": "string"}}}
    check("held-out inputs bind by position, not by our names",
          spec.bind(schema, ["27AAPFU0939F1Z"]), {"gstin": "27AAPFU0939F1Z"})
    check("a mismatched arity does not bind", spec.bind(schema, ["a", "b"]), None)
    check("the task's own value in the code is caught",
          bool(spec.hardcode_problem(
              "def run(**kwargs):\n    return 'V' if kwargs['p'] == '27AAPFU0939F1Z' "
              "else '?'\n", ["27AAPFU0939F1Z"], "V")), True)


def test_registry() -> None:
    """persist ablates the read, not the write."""
    print("registry")
    source = "def run(**kwargs):\n    return kwargs['n'] * 2\n"
    tool = registry.Tool(name="doubler", description="Double.", source=source,
                         parameters={"type": "object", "additionalProperties": False,
                                     "required": ["n"],
                                     "properties": {"n": {"type": "integer"}}},
                         task_id="t1", family="demo", digest=registry.digest_of(source))
    off = registry.Registry(persist=False, arm="selftest")
    off.begin_task("t1")
    off.add(tool)
    check("a written tool runs through the sandbox", off.dispatch("doubler", {"n": 21}), "42")
    off.begin_task("t2")
    check("persist off drops it on the next task", off.get("doubler"), None)
    on = registry.Registry(persist=True, arm="selftest")
    on.begin_task("t1")
    on.add(tool)
    on.begin_task("t2")
    check("persist on keeps it", on.get("doubler") is not None, True)
    check("an identical rewrite is a duplicate, not a failure",
          on.collision("doubler", source), "duplicate")
    check("a built-in name is refused", bool(registry.name_problem("calc")), True)


def test_ladder_and_gates() -> None:
    """One switch per rung, and a corpus nobody can shortcut."""
    print("ladder and corpus")
    check("the ladder imports clean", ladder.assert_single_switch(), None)
    check("the headline pair differs by exactly one switch",
          ladder.BY_NAME["persist"].policy.diff(ladder.BY_NAME["holdout"].policy),
          ("persist",))
    problems = [p for _, found in gates.run_all() for p in found]
    check("every corpus gate passes", problems, [])


def test_criterion() -> None:
    """The criterion separates a real effect from a small one."""
    print("criterion")
    from tools import check_report

    check("a real effect is MET", report.verdict(check_report.build())[0], True)
    check("a tenth of a saving is NOT MET",
          report.verdict(check_report.build(ratio=0.9))[0], False)
    check("rare reuse is NOT MET", report.verdict(check_report.build(reuse=0.3))[0], False)
    named = check_report.named(check_report.build(null_correct=True))
    check("a guessing null arm voids the run", named["the null arm cannot guess"], False)


def test_harness_wiring() -> None:
    """The whole run loop, with the one expensive call replaced by a stub."""
    print("harness wiring")
    calls: list[tuple[str, str]] = []

    async def fake_run_task(task, family, *, arm, policy, reg):
        calls.append((arm, task["id"]))
        return {"task_id": task["id"], "family": task["family"], "member": task["member"],
                "arm": arm, "answer": "x", "gold": task["answer"], "correct": True,
                "outcome": "correct", "rounds": 1, "llm_calls": 1, "tool_calls": 0,
                "prompt_tokens": 10, "completion_tokens": 10, "writes": 0,
                "registered": 0, "repairs": 0, "rejections": {}, "notes": [],
                "offered": [["calc"]], "called": [], "reused_from": "", "reused": False,
                "answering_tool": "", "registry": [], "stopped": "", "retried": 0}

    real, agent.run_task = agent.run_task, fake_run_task
    try:
        payload = asyncio.run(harness.run("smoke", "", resume=False))
        check("smoke runs three arms over four tasks", len(payload["results"]), 12)
        check("the headline arm runs first", calls[0][0], "persist")
        check("every arm walks the tasks in the same order",
              len({tuple(t for a, t in calls if a == arm)
                   for arm in {a for a, _ in calls}}), 1)
        # A second run with resume on must add nothing.
        again = asyncio.run(harness.run("smoke", "", resume=True))
        check("a resumed run repeats no work", len(again["results"]), 12)
        check("and makes no further calls", len(calls), 12)
    finally:
        agent.run_task = real


def main() -> int:
    """Run every free check and report once at the end."""
    for suite in (test_guard, test_sandbox, test_protocol, test_spec, test_registry,
                  test_ladder_and_gates, test_criterion, test_harness_wiring):
        suite()
    print(f"\n{'ALL PASS' if not FAILURES else f'{len(FAILURES)} FAILED: {FAILURES}'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
