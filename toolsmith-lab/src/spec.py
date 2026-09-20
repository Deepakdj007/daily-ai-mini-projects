"""The write_tool call: its schema, and everything checkable about a proposal.

Inputs:  the raw arguments of a write_tool call, and the task it was made for
Outputs: a parsed Proposal, or a reason the model can act on

Two of the five fields cross as JSON-encoded strings rather than objects, and
that is not laziness. Groq's strict mode requires additionalProperties false
and every property listed in required, which means it cannot express "an
arbitrary object" at all - and a JSON Schema for a tool nobody has written yet
is exactly that. Sending them as strings makes every leaf a string, so strict
decoding is satisfiable, and the parsing and validation happen here where a
failure becomes a sentence the model can read.

The source crosses as an ARRAY OF LINES, and that one is measured rather than
reasoned. Asked for a whole Python module as a single JSON string, gpt-oss
failed four times out of four with a hard 400 from Groq - "Failed to parse tool
call arguments as JSON" - because a forty-line module inside one escaped string
is a great deal of escaping to get exactly right while sampling. The same
request with one line per array element succeeded every time. Short strings are
easy to emit; one enormous one is not.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from typing import Any

MIN_TESTS = 2

WRITE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "write_tool",
        "strict": True,
        "description": (
            "Write a new Python tool and add it to your toolbox. Use this only when "
            "no tool you already have can do the job. The tool is checked before it "
            "is accepted; if it is rejected you are told why and may try again."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["name", "description", "parameters_json_schema", "source", "tests"],
            "properties": {
                "name": {"type": "string", "description":
                         "Lowercase, matching ^[a-z][a-z0-9_]{2,47}$, e.g. gstin_check."},
                "description": {"type": "string", "description":
                                "One sentence. This is what you will see in your tool list."},
                "parameters_json_schema": {"type": "string", "description":
                                           'A JSON Schema object serialised as a STRING. It must have '
                                           '"type":"object", "additionalProperties":false, and every '
                                           'key of "properties" also listed in "required".'},
                "source": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "The complete Python module, ONE LINE PER ARRAY ELEMENT, with "
                        "no newline characters inside an element. Keep the leading "
                        "spaces of each line - they are the indentation. It must "
                        "define run(**kwargs) and return a JSON-serialisable value. "
                        "Standard library only."),
                },
                "tests": {
                    "type": "array",
                    "description": "At least two. Work the expected values out by hand.",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["args_json", "expect_json"],
                        "properties": {
                            "args_json": {"type": "string", "description":
                                          'The keyword arguments as JSON, e.g. {"prefix": "27AA..."}.'},
                            "expect_json": {"type": "string", "description":
                                            'What run() should return, as JSON, e.g. "V" or 4.'},
                        },
                    },
                },
            },
        },
    },
}


# The name is submit_answer and not final_answer, which is not a preference.
# A tool named `final_answer` makes gpt-oss return an empty message after every
# tool result: it emits a call to a tool named `answer` instead, which Groq
# either drops silently or rejects with a 400 saying a tool "not in
# request.tools" was called. The same schema under the name `submit_answer` or
# `report` works every time. Something about `final_answer` collides with the
# harmony format the model was trained on; the fix costs one word.
ANSWER_TOOL = "submit_answer"
ANSWER_FIELD = "value"

FINAL_ANSWER: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": ANSWER_TOOL,
        "strict": True,
        "description": "Give your final answer to the task. Call this exactly once, last.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": [ANSWER_FIELD],
            "properties": {ANSWER_FIELD: {
                "type": "string",
                "description": "The answer alone, in the form the task asked for.",
            }},
        },
    },
}

# The control's tool. Deliberately the same contract as a written tool and the
# same guard and sandbox, so the only thing that differs from `write_tool` is
# that nothing is named, checked against hidden inputs, or kept. Code sent here
# may hard-code the task's own values - that is what throwaway code does, and
# it makes this arm's job easier rather than harder, which is what makes a
# finding against it worth having.
RUN_PYTHON: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "run_python",
        "strict": True,
        "description": (
            "Run a one-off Python snippet and see what it returns. Nothing is kept; "
            "if you need it again on a later task you will have to write it again."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["source"],
            "properties": {"source": {
                "type": "array",
                "items": {"type": "string"},
                "description": ("A module defining run(**kwargs), called with no "
                                "arguments. ONE LINE PER ARRAY ELEMENT, leading "
                                "spaces kept. Standard library only."),
            }},
        },
    },
}


@dataclass(frozen=True, slots=True)
class Proposal:
    """One parsed write_tool call."""

    name: str
    description: str
    schema: dict
    source: str
    tests: tuple[tuple[dict, Any], ...]


def join_source(value: object) -> str:
    """Lines back into a module.

    A plain string is accepted too. The array is what the model is asked for,
    but a model that sends one string anyway has done nothing wrong, and
    refusing it would spend a repair attempt on the transport rather than on
    the code.
    """
    if isinstance(value, (list, tuple)):
        return "\n".join(str(line).rstrip("\n") for line in value) + "\n"
    return str(value or "")


def parse(raw: dict) -> tuple[Proposal | None, str]:
    """Turn raw write_tool arguments into a Proposal, or say what is wrong."""
    name = str(raw.get("name", "")).strip()
    description = " ".join(str(raw.get("description", "")).split())
    source = join_source(raw.get("source"))
    try:
        schema = json.loads(raw.get("parameters_json_schema") or "{}")
    except json.JSONDecodeError as exc:
        return None, f"parameters_json_schema is not valid JSON: {exc}"
    if problem := schema_problem(schema):
        return None, problem

    tests: list[tuple[dict, Any]] = []
    for index, case in enumerate(raw.get("tests") or [], start=1):
        try:
            args = json.loads(case.get("args_json") or "{}")
            expect = json.loads(case.get("expect_json") or "null")
        except (json.JSONDecodeError, AttributeError) as exc:
            return None, f"test {index} is not valid JSON: {exc}"
        if not isinstance(args, dict):
            return None, f"test {index}: args_json must be a JSON object of keyword arguments."
        tests.append((args, expect))
    if len(tests) < MIN_TESTS:
        return None, (f"write at least {MIN_TESTS} tests. One test cannot tell a "
                      "correct implementation from one that happens to fit.")
    return Proposal(name, description, schema, source, tuple(tests)), ""


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
    required = schema.get("required")
    if not isinstance(required, list) or set(required) != set(properties):
        return ('every key of "properties" must also appear in "required" - strict '
                f'schemas have no optional fields. properties: {sorted(properties)}.')
    return ""


def required_order(schema: dict) -> list[str]:
    """The tool's own argument names, in the order it declared them."""
    return [key for key in schema.get("required", []) if key in schema.get("properties", {})]


def bind(schema: dict, values: list) -> dict | None:
    """Map positional family inputs onto whatever the tool called its arguments.

    A tool is free to call its parameter `gstin` where the family calls it
    `prefix`; insisting on our names would mean the held-out check measured
    whether the model guessed a keyword rather than whether its code
    generalises. Matching by position against the tool's own declared order is
    what keeps that measurement about the code.
    """
    names = required_order(schema)
    if len(names) != len(values):
        return None
    return dict(zip(names, values))


def _literals(value: object) -> list[str]:
    """Every scalar inside a task input, flattened to strings."""
    if isinstance(value, (list, tuple)):
        return [item for entry in value for item in _literals(entry)]
    return [str(value)]


def hardcode_problem(source: str, inputs: list, answer: object) -> str:
    """Whether this tool has the current task's input or answer written into it.

    A tool that contains the thing it was asked about is not a tool, it is an
    answer wearing a function's clothes. The held-out inputs would catch it
    too, but only on the arms that run them, and the message here is far more
    useful than a count of failures.
    """
    targets = [text for text in _literals(inputs) if len(text) >= 3]
    if isinstance(answer, str) and len(answer) >= 3:
        targets.append(answer)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""                       # the guard reports syntax, not this
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        for target in targets:
            if target in node.value:
                return (f"the value {target!r} from this task appears in your code on "
                        f"line {node.lineno}. Implement the rule, not this case - "
                        "your tool will be run on inputs you have not seen.")
    return ""


if __name__ == "__main__":
    assert schema_problem({"type": "object", "additionalProperties": False,
                           "required": ["n"], "properties": {"n": {"type": "integer"}}}) == ""
    assert "additionalProperties" in schema_problem(
        {"type": "object", "required": ["n"], "properties": {"n": {}}})
    assert "required" in schema_problem(
        {"type": "object", "additionalProperties": False, "required": [], "properties": {"n": {}}})
    assert "object" in schema_problem({"type": "string"})

    good = {"name": "doubler", "description": "Double it.",
            "parameters_json_schema": json.dumps(
                {"type": "object", "additionalProperties": False,
                 "required": ["n"], "properties": {"n": {"type": "integer"}}}),
            "source": "def run(**kwargs):\n    'x'\n    return kwargs['n'] * 2\n",
            "tests": [{"args_json": '{"n": 2}', "expect_json": "4"},
                      {"args_json": '{"n": 3}', "expect_json": "6"}]}
    proposal, why = parse(good)
    assert proposal is not None and why == "", why
    assert proposal.tests[0] == ({"n": 2}, 4)
    assert required_order(proposal.schema) == ["n"]
    assert bind(proposal.schema, [7]) == {"n": 7}
    assert bind(proposal.schema, [7, 8]) is None, "an arity mismatch must not bind"

    one_test = dict(good, tests=good["tests"][:1])
    assert parse(one_test)[0] is None and "at least 2" in parse(one_test)[1]

    # A tool whose own parameter name differs from the family's must still bind.
    other = {"type": "object", "additionalProperties": False,
             "required": ["gstin"], "properties": {"gstin": {"type": "string"}}}
    assert bind(other, ["27AAPFU0939F1Z"]) == {"gstin": "27AAPFU0939F1Z"}

    cheat = "def run(**kwargs):\n    'x'\n    return 'V' if kwargs['p'] == '27AAPFU0939F1Z' else '?'\n"
    assert "27AAPFU0939F1Z" in hardcode_problem(cheat, ["27AAPFU0939F1Z"], "V")
    honest = "def run(**kwargs):\n    'x'\n    return kwargs['p'][-1]\n"
    assert hardcode_problem(honest, ["27AAPFU0939F1Z"], "V") == ""
    # The words family legitimately contains 'lakh' and 'thousand'; only the
    # whole answer counts as a cheat.
    words = "def run(**kwargs):\n    'x'\n    return 'lakh thousand'\n"
    assert hardcode_problem(words, [1234567], "twelve lakh thirty four thousand") == ""
    print("OK - spec")
