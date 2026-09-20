"""The three tools the agent starts with, and their schemas.

Inputs:  arguments from a model tool call
Outputs: a string for the model to read

Three, chosen so the baseline is a fair floor rather than a straw man. A
calculator, today's date and a text summariser are what a general assistant
plausibly ships with, and between them they look relevant to every family in
the corpus while solving none of it.

That last part is not a hope, it is a gate. `calc` evaluates one arithmetic
expression, so a family whose answer is one arithmetic expression would be
solved by the baseline and would stop measuring anything - which is why the
loan family asks for interest summed over a rounded schedule rather than the
EMI, and why src/gates.py checks the claim rather than trusting it.
"""

from __future__ import annotations

import ast
import operator
from typing import Any, Callable

from src import config

_OPERATORS: dict[type, Callable] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float:
    """Evaluate one arithmetic node, refusing anything that is not arithmetic.

    No bare eval(). The same default-deny walk the guard uses on whole modules,
    over a much smaller grammar.
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
    """Today's date, frozen so the corpus answers the same thing every run."""
    return config.FROZEN_TODAY


def text_stats(text: str) -> str:
    """Length and character-class counts for a string."""
    return (f"length={len(text)} digits={sum(c.isdigit() for c in text)} "
            f"letters={sum(c.isalpha() for c in text)} "
            f"upper={sum(c.isupper() for c in text)} "
            f"words={len(text.split())}")


FIXED: dict[str, Callable[..., str]] = {
    "calc": calc, "today": today, "text_stats": text_stats,
}

SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "calc",
            "description": "Evaluate one arithmetic expression, such as '1000000 * 0.0071'.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["expression"],
                "properties": {"expression": {
                    "type": "string",
                    "description": "An expression using + - * / // % ** and parentheses.",
                }},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "today",
            "description": "Today's date as an ISO string.",
            "parameters": {"type": "object", "additionalProperties": False,
                           "required": [], "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "text_stats",
            "description": "Length and character-class counts for a piece of text.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text"],
                "properties": {"text": {"type": "string"}},
            },
        },
    },
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
    except Exception as exc:  # noqa: BLE001 - hand every failure back as text
        return f"ERROR: {type(exc).__name__}: {exc}"


if __name__ == "__main__":
    assert calc("128 - 47 - 39") == "42"
    assert calc("2 ** 10") == "1024"
    assert calc("1 / 0").startswith("error:")
    assert calc("__import__('os')").startswith("error:")
    assert calc("open('x')").startswith("error:")
    assert today() == config.FROZEN_TODAY
    assert "length=15" in text_stats("27AAPFU0939F1ZV")
    assert "digits=7" in text_stats("27AAPFU0939F1ZV")
    assert call("nope", {}).startswith("ERROR: there is no tool")
    assert call("calc", {"wrong": 1}).startswith("ERROR: wrong arguments")
    assert {s["function"]["name"] for s in SCHEMAS} == set(FIXED), "schemas must match"
    print(f"fixed tools: {', '.join(FIXED)}")
    print(f"  calc('1000000 * 0.0070833')  -> {calc('1000000 * 0.0070833')}")
    print(f"  text_stats('27AAPFU0939F1Z') -> {text_stats('27AAPFU0939F1Z')}")
    print("OK - tools")
