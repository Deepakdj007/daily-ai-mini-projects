"""Turning a write_tool call into a registered tool, or a reason it is not.

Inputs:  a write_tool call and the task it was made for
Outputs: an Outcome carrying the tool, or the exact text sent back to the model

Two decisions in the schema below are measured rather than reasoned.

The source crosses as an ARRAY OF LINES. Asked for a whole Python module as
one JSON string, gpt-oss failed four times out of four with a hard 400 from
Groq - a forty-line module inside one escaped string is a lot of escaping to
get exactly right while sampling. One line per array element worked every time.

The JSON Schema for the new tool crosses as a STRING, because Groq's strict
mode requires additionalProperties false and every key in required, which
means it cannot express "an arbitrary object" at all - and a schema for a tool
nobody has written yet is exactly that.

One message here is deliberately vague, and it is the most important one. When
a tool passes its own tests and fails the held-out inputs, the model is told
how many failed and nothing else. Naming them would hand it the test set, and
the check would stop measuring whether the code generalises.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, replace
from typing import Any

from src import config, guard, registry, sandbox

MIN_TESTS = 2

WRITE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "write_tool",
        "strict": True,
        "description": (
            "Write a new Python tool and add it to your toolbox. Use this only when "
            "no tool you already have can do the job. The tool is checked before it "
            "is accepted; if it is rejected you are told why and may try again."),
        "parameters": {
            "type": "object", "additionalProperties": False,
            "required": ["name", "description", "parameters_json_schema", "source", "tests"],
            "properties": {
                "name": {"type": "string", "description":
                         "Lowercase, matching ^[a-z][a-z0-9_]{2,47}$, e.g. gstin_check."},
                "description": {"type": "string", "description":
                                "One sentence. This is what you will see in your tool list."},
                "parameters_json_schema": {"type": "string", "description":
                                           'A JSON Schema object serialised as a STRING. It needs '
                                           '"type":"object", "additionalProperties":false, and every '
                                           'key of "properties" also listed in "required".'},
                "source": {"type": "array", "items": {"type": "string"}, "description":
                           "The complete Python module, ONE LINE PER ARRAY ELEMENT, with no "
                           "newline characters inside an element. Keep the leading spaces - "
                           "they are the indentation. It must define run(**kwargs) and return "
                           "a JSON-serialisable value."},
                "tests": {"type": "array", "description":
                          "At least two. Work the expected values out by hand from the rule.",
                          "items": {"type": "object", "additionalProperties": False,
                                    "required": ["args_json", "expect_json"],
                                    "properties": {
                                        "args_json": {"type": "string"},
                                        "expect_json": {"type": "string"}}}},
            },
        },
    },
}

# The name is submit_answer and not final_answer, which is not a preference. A
# tool named `final_answer` makes gpt-oss return an empty message after every
# tool result: it emits a call to a tool named `answer` instead, which Groq
# either drops silently or rejects with a 400. The same schema named
# submit_answer works every time.
ANSWER_TOOL, ANSWER_FIELD = "submit_answer", "value"

SUBMIT_ANSWER: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": ANSWER_TOOL, "strict": True,
        "description": "Give your final answer. Call this exactly once, last.",
        "parameters": {"type": "object", "additionalProperties": False,
                       "required": [ANSWER_FIELD],
                       "properties": {ANSWER_FIELD: {
                           "type": "string",
                           "description": "The answer alone, in the form the task asked for."}}},
    },
}


@dataclass(frozen=True, slots=True)
class Outcome:
    """What happened to one write_tool call."""

    stage: str      # registered|duplicate|schema|name|collision|hardcode|guard|
                    # contract|selftest|holdout|timeout
    ok: bool
    message: str
    tool: registry.Tool | None = None

    @property
    def burns_repair(self) -> bool:
        """A duplicate is a no-op, not a failure, so it costs no attempt."""
        return not self.ok and self.stage != "duplicate"


def join_source(value: object) -> str:
    """Lines back into a module. A plain string is accepted too."""
    if isinstance(value, (list, tuple)):
        return "\n".join(str(line).rstrip("\n") for line in value) + "\n"
    return str(value or "")


def schema_problem(schema: object) -> str:
    """Why this parameter schema will not be accepted, or an empty string."""
    if not isinstance(schema, dict):
        return "parameters_json_schema must be a JSON object."
    if schema.get("type") != "object":
        return 'parameters_json_schema needs "type": "object" at the top level.'
    if schema.get("additionalProperties") is not False:
        return 'parameters_json_schema needs "additionalProperties": false.'
    properties = schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        return 'parameters_json_schema needs a non-empty "properties" object.'
    if set(schema.get("required") or []) != set(properties):
        return ('every key of "properties" must also appear in "required" - '
                f'strict schemas have no optional fields. Got {sorted(properties)}.')
    return ""


def bind(schema: dict, values: list) -> dict | None:
    """Map held-out inputs onto whatever the tool called its arguments.

    A tool may call its parameter `gstin` where the task calls it `prefix`.
    Insisting on our names would measure whether the model guessed a keyword
    rather than whether its code generalises, so binding is positional against
    the tool's own declared order.
    """
    names = [k for k in schema.get("required", []) if k in schema.get("properties", {})]
    return dict(zip(names, values)) if len(names) == len(values) else None


def holdout_calls(schema: dict, items: list) -> tuple[list[dict], bool]:
    """How to run one held-out batch against this tool: (calls, join the results).

    The model owns the interface, so the check has to adapt to it. Asked for
    the check characters of four prefixes, a model may write one tool that
    takes the whole list, or one that takes a single prefix and expects to be
    called four times. Both are correct tools. Deciding from the declared type
    of its own parameter - array or not - means the held-out check runs either
    way, instead of quietly not running against half the tools that get
    written.
    """
    names = [k for k in schema.get("required", []) if k in schema.get("properties", {})]
    if len(names) != 1:
        return [], False
    if schema["properties"][names[0]].get("type") == "array":
        return [{names[0]: items}], False              # one call, the whole list
    return [{names[0]: item} for item in items], True   # one call per item


def hardcode_problem(source: str, prompt: str) -> str:
    """Whether the tool has this task's own input written into it.

    A tool containing the thing it was asked about is not a tool, it is an
    answer wearing a function's clothes.
    """
    # Only words that look like DATA: long, and carrying a digit. Length alone
    # is not enough - "prefixes" is eight characters, appears in the prompt as
    # ordinary English, and is also the obvious name for the parameter, so a
    # length-only rule rejected a well-named tool and spent a repair attempt
    # on it. Every input in this corpus is alphanumeric: 16TEUYJ4263R1Z,
    # 2025-03-17. Ordinary prose words are not.
    words = {word for word in (w.strip(".,?:;!'\"()") for w in prompt.split())
             if len(word) >= 8 and any(ch.isdigit() for ch in word)}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""                       # the guard reports syntax, not this
    # Docstrings are exempt. Quoting the task's worked example in a docstring
    # is documentation, not a special case baked into the logic, and rejecting
    # it costs a repair attempt for writing a clear comment.
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.FunctionDef)) and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in docstrings):
            for word in words:
                if word in node.value:
                    return (f"the value {word!r} from this task appears in your "
                            f"code on line {node.lineno}. Implement the rule, not "
                            "this case - your tool will be run on inputs you have "
                            "not seen.")
    return ""


def attempt(raw: dict, *, task: dict, family: dict, reg: registry.Registry) -> Outcome:
    """Check one proposed tool and register it if every check passes."""
    name = str(raw.get("name", "")).strip()
    source = join_source(raw.get("source"))
    try:
        schema = json.loads(raw.get("parameters_json_schema") or "{}")
    except json.JSONDecodeError as exc:
        return Outcome("schema", False, f"REJECTED (schema): not valid JSON: {exc}")
    if problem := schema_problem(schema):
        return Outcome("schema", False, f"REJECTED (schema): {problem}")

    tests: list[tuple[dict, Any]] = []
    for index, case in enumerate(raw.get("tests") or [], start=1):
        try:
            args = json.loads(case.get("args_json") or "{}")
            expect = json.loads(case.get("expect_json") or "null")
        except (json.JSONDecodeError, AttributeError) as exc:
            return Outcome("schema", False, f"REJECTED (schema): test {index}: {exc}")
        tests.append((args, expect))
    if len(tests) < MIN_TESTS:
        return Outcome("schema", False, f"REJECTED (schema): write at least "
                                        f"{MIN_TESTS} tests.")

    if problem := registry.name_problem(name):
        return Outcome("name", False, f"REJECTED (name): {problem}")
    clash = reg.collision(name, source)
    if clash == "duplicate":
        return Outcome("duplicate", False, f"ALREADY REGISTERED: '{name}' with "
                                           "identical code is in your tool list. Call it.")
    if clash:
        return Outcome("collision", False, f"REJECTED (name): {clash}")
    if problem := hardcode_problem(source, task["prompt"]):
        return Outcome("hardcode", False, f"REJECTED (hardcoded): {problem}")

    verdict = guard.guard(source)
    if not verdict.ok:
        return Outcome("guard", False, f"REJECTED (guard, {verdict.layer}): "
                                       f"{verdict.reason} Use only the standard "
                                       "library modules listed in your instructions.")

    # The model's own tests and the held-out inputs share ONE sandbox process.
    held = family.get("holdout", [])
    plans = [holdout_calls(schema, row["inputs"]) for row in held]
    runnable = bool(held) and all(plan for plan, _ in plans)
    calls = [args for args, _ in tests]
    if runnable:
        for plan, _ in plans:
            calls.extend(plan)
    results, raw_run = sandbox.execute(source, calls)

    if raw_run.outcome in ("timeout", "output"):
        return Outcome("timeout", False, f"FAILED ({raw_run.outcome}): your checks "
                                         "did not finish cleanly. Remove any loop "
                                         "that may not terminate.")
    first = results[0]
    if not first.get("ok") and "name 'run' is not defined" in str(first.get("error", "")):
        return Outcome("contract", False, "REJECTED (contract): the module must "
                                          f"define run(**kwargs). Got: {first.get('error')}")

    own = results[:len(tests)]
    passed = sum(r.get("ok") and r.get("value") == expect
                 for r, (_, expect) in zip(own, tests))
    if passed < len(tests):
        detail = ""
        for (args, expect), row in zip(tests, own):
            if not (row.get("ok") and row.get("value") == expect):
                got = row.get("value") if row.get("ok") else f"raised {row.get('error')}"
                detail = (f"\n  args {json.dumps(args)} -> you expected "
                          f"{json.dumps(expect)}, the code returned "
                          f"{json.dumps(got, default=str)}")
                break
        return Outcome("selftest", False,
                       f"FAILED (your own tests): {len(tests) - passed} of "
                       f"{len(tests)} disagreed.{detail}\nOne of those two is "
                       "wrong. Work out which by hand before rewriting. Do not "
                       "delete or weaken the failing test.")

    held_pass = 0
    if runnable:
        cursor = len(tests)
        for (plan, joined), row in zip(plans, held):
            chunk = results[cursor:cursor + len(plan)]
            cursor += len(plan)
            if not all(r.get("ok") for r in chunk):
                continue
            got = ("".join(str(r["value"]) for r in chunk) if joined
                   else str(chunk[0]["value"]))
            held_pass += got.strip().upper() == str(row["expect"]).strip().upper()
        if held_pass < len(held):
            # The hint names the KIND of case that failed and never an input.
            # Counts alone are not actionable - the model spends every repair
            # attempt guessing - and naming the inputs would hand over the test
            # set, which is the one thing that would stop this measuring
            # whether the code generalises. A category is what a good error
            # message gives you, and it is what a real specification already
            # warned about.
            hint = family.get("holdout_hint", "")
            return Outcome("holdout", False,
                           f"FAILED (held-out check): your code passes all {passed} "
                           f"of your own tests but fails {len(held) - held_pass} of "
                           f"{len(held)} hidden inputs of the same kind. The inputs "
                           "themselves are not shown, but "
                           + (hint or "they cover cases your examples did not")
                           + ". Re-read the rule and handle that case, rather than "
                             "shaping the code around the examples you were given.")

    tool = registry.Tool(name=name, description=str(raw.get("description", "")).strip(),
                         parameters=schema, source=source, task_id=task["id"],
                         tests_passed=passed, holdout_passed=held_pass,
                         digest=registry.digest_of(source))
    reg.add(tool)
    extra = f" and {held_pass} held-out checks" if runnable else ""
    return Outcome("registered", True, tool=tool, message=(
        f"REGISTERED '{name}' - it passed all {passed} of your tests{extra}. It is "
        "in your tool list from the next message - call it now."))


def refusal() -> str:
    """What the model is told once its repair attempts are spent."""
    return (f"REFUSED: {config.MAX_REPAIR_ATTEMPTS} attempts at writing a tool have "
            "failed on this task. No more tools will be accepted. Answer with what "
            "you have, or say that you cannot.")


if __name__ == "__main__":
    assert join_source(["def run(**kwargs):", "    return 1"]) == \
        "def run(**kwargs):\n    return 1\n"
    assert schema_problem({"type": "object", "additionalProperties": False,
                           "required": ["n"], "properties": {"n": {}}}) == ""
    assert "additionalProperties" in schema_problem(
        {"type": "object", "required": ["n"], "properties": {"n": {}}})
    other = {"type": "object", "additionalProperties": False,
             "required": ["gstin"], "properties": {"gstin": {"type": "string"}}}
    assert bind(other, ["27AAPFU0939F1Z"]) == {"gstin": "27AAPFU0939F1Z"}
    assert bind(other, ["a", "b"]) is None
    # A per-item tool is called once per item and its answers joined; a tool
    # declaring an array parameter is called once with the whole list.
    calls, joined = holdout_calls(other, ["A", "B"])
    assert calls == [{"gstin": "A"}, {"gstin": "B"}] and joined is True
    batch = {"type": "object", "additionalProperties": False, "required": ["items"],
             "properties": {"items": {"type": "array"}}}
    calls, joined = holdout_calls(batch, ["A", "B"])
    assert calls == [{"items": ["A", "B"]}] and joined is False
    cheat = ("def run(**kwargs):\n"
             "    return 'V' if kwargs['p'] == '27AAPFU0939F1Z' else '?'\n")
    assert "27AAPFU0939F1Z" in hardcode_problem(
        cheat, "check character for 27AAPFU0939F1Z please")
    assert hardcode_problem("def run(**kwargs):\n    return kwargs['p'][-1]\n",
                            "check character for 27AAPFU0939F1Z please") == ""
    # An ordinary English word from the prompt that is also the obvious
    # parameter name must NOT be mistaken for a hard-coded input.
    named = ("def run(**kwargs):\n"
             "    return ''.join(p[-1] for p in kwargs['prefixes'])\n")
    assert hardcode_problem(
        named, "What are the check characters for these GSTIN prefixes: "
               "16TEUYJ4263R1Z, 14ELRKY8914Q4Z?") == "", "prose words are not data"
    # And a real input still is.
    baked = "def run(**kwargs):\n    return {'16TEUYJ4263R1Z': 'K'}[kwargs['p']]\n"
    assert "16TEUYJ4263R1Z" in hardcode_problem(
        baked, "these GSTIN prefixes: 16TEUYJ4263R1Z, 14ELRKY8914Q4Z?")
    print("OK - smith")
