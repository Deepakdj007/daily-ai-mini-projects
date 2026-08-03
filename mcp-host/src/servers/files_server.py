"""
MCP server exposing sandboxed file operations.

Every caller-supplied path is resolved against one root directory. Anything that
resolves outside it is rejected, so the model cannot read C:\\Windows or write
over your source. Standalone by design: it imports nothing from src/, so the
host can launch it as a plain child process with no PYTHONPATH.

Inputs: --root <dir>, then MCP requests on stdin.
Outputs: read_file, write_file, list_files tools on stdout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mcp.server import MCPServer

mcp = MCPServer("files")

# Replaced in main() before the server starts serving.
ROOT: Path = Path.cwd()


def _resolve(path: str) -> Path:
    """
    Resolve a caller-supplied relative path inside ROOT.

    Raises ValueError if the result escapes the sandbox, which covers '..',
    absolute paths, and symlinks that point outside.
    """
    candidate = (ROOT / path).resolve()
    if candidate != ROOT and ROOT not in candidate.parents:
        raise ValueError(f"path escapes the sandbox root: {path!r}")
    return candidate


@mcp.tool()
def read_file(path: str) -> str:
    """Read a UTF-8 text file. Path is relative to the sandbox root, e.g. 'notes/todo.md'."""
    target = _resolve(path)
    if not target.is_file():
        raise ValueError(f"no such file: {path!r}")
    return target.read_text(encoding="utf-8")


@mcp.tool()
def write_file(path: str, content: str) -> str:
    """Write UTF-8 text to a file, creating parent folders as needed. Path is relative
    to the sandbox root: pass 'summary.md', not 'workspace/summary.md'."""
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} chars to {target.relative_to(ROOT).as_posix()}"


@mcp.tool()
def list_files(subdir: str = "") -> list[str]:
    """List every file under the sandbox root, recursively. Pass subdir to narrow it."""
    base = _resolve(subdir) if subdir else ROOT
    if not base.is_dir():
        raise ValueError(f"no such directory: {subdir!r}")
    return sorted(p.relative_to(ROOT).as_posix() for p in base.rglob("*") if p.is_file())


def main() -> None:
    """Parse --root, pin the sandbox, and serve MCP over stdio."""
    global ROOT
    parser = argparse.ArgumentParser(description="Sandboxed filesystem MCP server")
    parser.add_argument("--root", required=True, help="the only directory this server may touch")
    args = parser.parse_args()

    ROOT = Path(args.root).resolve()
    ROOT.mkdir(parents=True, exist_ok=True)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
