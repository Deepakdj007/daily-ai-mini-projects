"""The one tool the agent under test can call: a safe arithmetic calculator.

Inputs: an arithmetic expression string (e.g. "128 - 47 - 39").
Outputs: the OpenAI-style tool schema, plus a runner that evaluates the
expression using Python's ast module so no arbitrary code can execute.
"""

import ast
import operator
from typing import Any

_OPERATORS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
}

CALCULATOR_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a basic arithmetic expression, e.g. '128 - 47 - 39' or '960 / 8'.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "An arithmetic expression using +, -, *, /, %, ** and parentheses.",
                }
            },
            "required": ["expression"],
        },
    },
}


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate a parsed arithmetic AST node."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


def run_calculator(expression: str) -> str:
    """Safely evaluate an arithmetic expression string and return the result as text."""
    try:
        parsed = ast.parse(expression, mode="eval")
        result = _eval_node(parsed.body)
        return str(result)
    except Exception as exc:
        return f"error: could not evaluate '{expression}' ({exc})"
