"""
Turns five independent servers into one legible tool inventory.

Namespacing so two servers can both expose a 'search' tool, a per-server status
record for the UI, the allowlist that decides what the model actually sees, and
a filter for the startup noise the capability probe generates.

Inputs: tool names and Implementation info from the session group.
Outputs: namespaced names, ServerStatus records, filtered tool lists.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import mcp.types as types

from src.registry import ServerSpec

# Separates server key from tool name. Not a dot: Groq function names must match
# ^[a-zA-Z0-9_-]{1,64}$, which rejects dots.
NAME_SEP = "__"


@dataclass
class ServerStatus:
    """What happened when we tried to connect one server, for the UI to render."""

    key: str
    connected: bool
    error: str = ""
    tools: list[str] = field(default_factory=list)
    exposed: list[str] = field(default_factory=list)


class Namer:
    """
    Namespaces tools as ``serverkey__toolname``.

    The SDK hands the hook the server's own advertised name ('mcp-server-git'),
    not our servers.json key, so we set `current` before each connect. Memoising
    on the advertised name keeps this correct if a server later re-announces its
    tools and the group re-aggregates against a stale `current`.
    """

    def __init__(self) -> None:
        self.current = ""
        self._seen: dict[str, str] = {}

    def __call__(self, name: str, server_info: types.Implementation) -> str:
        key = self._seen.setdefault(server_info.name, self.current)
        return f"{key}{NAME_SEP}{name}"


class CapabilityNoiseFilter(logging.Filter):
    """
    Drops 'Could not fetch prompts/resources' warnings.

    The session group probes every capability on connect, and a server that
    implements only tools logs a warning for each one it lacks. That is normal
    and would otherwise print ten alarming lines on a healthy startup.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "Could not fetch" not in record.getMessage()


def apply_allowlist(namespaced: list[str], spec: ServerSpec) -> list[str]:
    """
    Filter a server's tools down to its configured allowlist, if it has one.

    Every schema is resent on every turn, so this is the difference between a
    five-server host that fits the free tier and one that does not.
    """
    if not spec.tool_allowlist:
        return list(namespaced)
    wanted = {f"{spec.key}{NAME_SEP}{tool}" for tool in spec.tool_allowlist}
    return [name for name in namespaced if name in wanted]
