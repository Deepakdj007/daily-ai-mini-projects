"""Six zero-network tools the agent can call.

Each tool is a plain Python function with typed args and a docstring — the Ollama SDK
converts these into the JSON schema the model sees. File tools are locked to SANDBOX_DIR.
"""

import ast
import operator
from datetime import datetime
from pathlib import Path

from src.config import SANDBOX_DIR

SANDBOX_DIR.mkdir(parents=True, exist_ok=True)

# Operators the calculator is allowed to evaluate. Anything else is rejected.
_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate one AST node, allowing only numbers and arithmetic ops."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


def _safe_path(filename: str) -> Path:
    """Resolve filename inside SANDBOX_DIR, rejecting traversal and absolute paths."""
    candidate = (SANDBOX_DIR / filename).resolve()
    if not candidate.is_relative_to(SANDBOX_DIR.resolve()):
        raise ValueError(f"Refused: '{filename}' escapes the sandbox folder.")
    return candidate


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression and return the result.

    Args:
      expression: A math expression using numbers and + - * / ** % and parentheses,
        for example '0.15 * 1240'.
    Returns:
      The numeric result as a string.
    """
    result = _eval_node(ast.parse(expression, mode="eval").body)
    return str(result)


def write_note(filename: str, content: str) -> str:
    """Save text content to a named note file.

    Args:
      filename: Name of the note file, for example 'tip.txt'.
      content: The text to store in the note.
    Returns:
      A confirmation message with the saved path.
    """
    path = _safe_path(filename)
    path.write_text(content, encoding="utf-8")
    return f"Saved note '{filename}' ({len(content)} characters)."


def read_note(filename: str) -> str:
    """Read back the content of a named note file.

    Args:
      filename: Name of the note file to read, for example 'tip.txt'.
    Returns:
      The note's text content.
    """
    path = _safe_path(filename)
    if not path.exists():
        return f"No note named '{filename}' exists."
    return path.read_text(encoding="utf-8")


def list_notes() -> str:
    """List the names of all saved note files.

    Returns:
      A comma-separated list of note filenames, or a message if none exist.
    """
    names = sorted(p.name for p in SANDBOX_DIR.iterdir() if p.is_file())
    if not names:
        return "No notes saved yet."
    return ", ".join(names)


def current_datetime() -> str:
    """Get the current local date and time.

    Returns:
      The current date and time, for example '2026-07-11 14:30'.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def word_count(text: str) -> str:
    """Count the words and characters in a piece of text.

    Args:
      text: The text to measure.
    Returns:
      The word and character counts.
    """
    return f"{len(text.split())} words, {len(text)} characters."


TOOL_FUNCTIONS = [calculator, write_note, read_note, list_notes, current_datetime, word_count]
TOOL_MAP = {fn.__name__: fn for fn in TOOL_FUNCTIONS}
