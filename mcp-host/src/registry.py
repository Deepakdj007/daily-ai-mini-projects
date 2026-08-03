"""
Turns servers.json into launch parameters the MCP client can use.

Handles the three things that bite on a real machine: relative paths that depend
on where you launched from, the child-process environment allowlist, and picking
the right Python interpreter for the servers we wrote ourselves.

Inputs: servers.json.
Outputs: a list of ServerSpec, each holding StdioServerParameters plus metadata.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from mcp import StdioServerParameters
from mcp.client.stdio import get_default_environment

from src import config

# File suffixes that mark an argument as a path we own rather than a package name.
_PATH_SUFFIXES = (".py", ".db", ".json", ".sqlite")


@dataclass
class ServerSpec:
    """One configured MCP server: how to launch it and which tools to expose."""

    key: str
    params: StdioServerParameters
    tool_allowlist: list[str] = field(default_factory=list)


def _resolve_command(command: str) -> str:
    """
    Map a config command onto something launchable.

    'python' becomes this interpreter, so our own servers run in the project venv
    without nesting `uv run` inside `uv run`. Everything else is left alone for
    the SDK's Windows resolver, which already tries .cmd/.bat/.exe/.ps1.
    """
    return sys.executable if command == "python" else command


def _is_path_arg(arg: str) -> bool:
    """
    Decide whether an argument names a file or directory we should make absolute.

    Flags and package names are left alone; '.', anything containing a slash, and
    known file suffixes are treated as paths.
    """
    if arg.startswith("-"):
        return False
    return arg in (".", "..") or "/" in arg or "\\" in arg or arg.endswith(_PATH_SUFFIXES)


def _resolve_arg(arg: str) -> str:
    """Make a path argument absolute against the project root, leaving flags alone."""
    if not _is_path_arg(arg):
        return arg
    return str((config.PROJECT_ROOT / arg).resolve())


def _resolve_cwd(cwd: str | None) -> str:
    """
    Pick the working directory a server starts in, defaulting to the project root.

    This matters more than it looks. Several servers take a path argument per call
    (mcp-server-git wants repo_path on every tool), and the model naturally passes
    '.', which the server resolves against its own cwd.
    """
    if not cwd:
        return str(config.PROJECT_ROOT)
    return str((config.PROJECT_ROOT / cwd).resolve())


def _build_env(overrides: dict[str, str] | None) -> dict[str, str]:
    """
    Merge per-server env over the SDK's allowlisted default.

    Child processes do NOT inherit your environment. On Windows the SDK passes
    only PATH, APPDATA, TEMP and a handful of others, so anything a server needs
    has to be named here explicitly.
    """
    env = get_default_environment()
    env.update(overrides or {})
    return env


def load_servers(path: Path | None = None) -> list[ServerSpec]:
    """Read servers.json and return one ServerSpec per configured server."""
    source = path or config.SERVERS_JSON
    if not source.is_file():
        raise RuntimeError(f"missing server config: {source}")

    raw = json.loads(source.read_text(encoding="utf-8"))
    entries: dict[str, dict] = raw.get("mcpServers", {})
    if not entries:
        raise RuntimeError(f"{source} has no entries under 'mcpServers'")

    specs: list[ServerSpec] = []
    for key, entry in entries.items():
        params = StdioServerParameters(
            command=_resolve_command(entry["command"]),
            args=[_resolve_arg(a) for a in entry.get("args", [])],
            env=_build_env(entry.get("env")),
            cwd=_resolve_cwd(entry.get("cwd")),
            # Windows consoles hand back bytes that aren't valid UTF-8 often enough
            # that strict decoding will kill a session mid-run.
            encoding_error_handler="replace",
        )
        specs.append(ServerSpec(key=key, params=params, tool_allowlist=entry.get("tools", [])))

    return specs
