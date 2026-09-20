"""The two tools the agent starts with, and their schemas.

Inputs:  arguments from a model tool call
Outputs: a string for the model to read

Two, chosen so the starting toolbox is a fair one rather than a straw man. A
calculator and today's date are what a general assistant plausibly ships with,
and between them they look relevant to the demo tasks while solving neither:
`calc` evaluates one arithmetic expression, and a checksum is a rule applied
across fourteen characters, eight times over.
"""

from __future__ import annotations

import ast
import operator
from datetime import date
from typing import Any, Callable

_OPERATORS: dict[type, Callable] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float:
    """Evaluate one arithmetic node, refusing anything that is not arithmetic.

    No bare eval(). This is the same default-deny walk the guard uses on whole
    modules, over a much smaller grammar.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"not arithmetic: {ast.dump(node)[:60]}")


def calc(expression: str) -> str:
    """Evaluate one arithmetic expression."""
    try:
        value = _eval_node(ast.parse(expression, mode="eval").body)
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError) as exc:
        return f"error: could not evaluate {expression!r} ({exc})"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return f"{value:.10g}" if isinstance(value, float) else str(value)


def today() -> str:
    """Today's date, as an ISO string."""
    return date.today().isoformat()


FIXED: dict[str, Callable[..., str]] = {"calc": calc, "today": today}

SCHEMAS: list[dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "calc",
        "description": "Evaluate one arithmetic expression, such as '36 - 221 % 36'.",
        "parameters": {"type": "object", "additionalProperties": False,
                       "required": ["expression"],
                       "properties": {"expression": {
                           "type": "string",
                           "description": "An expression using + - * / // % ** and parentheses."}}}}},
    {"type": "function", "function": {
        "name": "today",
        "description": "Today's date as an ISO string.",
        "parameters": {"type": "object", "additionalProperties": False,
                       "required": [], "properties": {}}}},
]


def call(name: str, args: dict) -> str:
    """Run one fixed tool. Never raises: a failure is text the model can read."""
    fn = FIXED.get(name)
    if fn is None:
        return f"ERROR: there is no tool called {name!r}."
    try:
        return str(fn(**args))
    except TypeError as exc:
        return f"ERROR: wrong arguments for {name}: {exc}"
    except Exception as exc:              # noqa: BLE001 - hand it back as text
        return f"ERROR: {type(exc).__name__}: {exc}"


if __name__ == "__main__":
    assert calc("128 - 47 - 39") == "42"
    assert calc("(36 - 221 % 36) % 36") == "31"
    assert calc("1 / 0").startswith("error:")
    assert calc("__import__('os')").startswith("error:")
    assert calc("open('x')").startswith("error:")
    assert call("nope", {}).startswith("ERROR: there is no tool")
    assert call("calc", {"wrong": 1}).startswith("ERROR: wrong arguments")
    assert {s["function"]["name"] for s in SCHEMAS} == set(FIXED)
    print(f"fixed tools: {', '.join(FIXED)}")
    print(f"  calc('(36 - 221 % 36) % 36') -> {calc('(36 - 221 % 36) % 36')}")
    print("OK - tools")
