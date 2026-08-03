"""
The MCP host: keeps five servers connected and lets synchronous code call them.

Streamlit re-runs its script on every interaction, but ClientSessionGroup owns
five live subprocesses inside anyio task groups bound to the loop that created
them. So the group lives on a daemon thread with its own permanent loop, parked
on an asyncio.Event, and callers submit work with run_coroutine_threadsafe.

Inputs: ServerSpec list from registry. Outputs: MCPHost.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any, Awaitable, TypeVar

import mcp.types as types
from mcp import ClientSessionGroup

from src import config
from src.inventory import CapabilityNoiseFilter, Namer, ServerStatus, apply_allowlist
from src.registry import ServerSpec, load_servers

T = TypeVar("T")


class MCPHost:
    """Owns a background event loop with every configured MCP server connected to it."""

    def __init__(self, specs: list[ServerSpec] | None = None) -> None:
        self.specs = specs if specs is not None else load_servers()
        self.status: list[ServerStatus] = []
        self._group: ClientSessionGroup | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._shutdown: asyncio.Event | None = None
        self._ready = threading.Event()
        self._boot_error: BaseException | None = None

    # ----- lifecycle -------------------------------------------------------

    def start(self) -> None:
        """Boot the loop thread and block until every server has been tried."""
        logging.getLogger().addFilter(CapabilityNoiseFilter())
        self._thread = threading.Thread(target=self._run_loop, name="mcp-host", daemon=True)
        self._thread.start()

        if not self._ready.wait(timeout=config.CONNECT_TIMEOUT):
            raise RuntimeError(
                f"MCP servers did not come up within {config.CONNECT_TIMEOUT:.0f}s. First run "
                "downloads the uvx servers — try the pre-warm commands in the README."
            )
        if self._boot_error is not None:
            raise RuntimeError(f"MCP host failed to start: {self._boot_error}")

    def stop(self) -> None:
        """
        Unwind the thread and terminate the children. Dropping the reference is not
        enough — without this, every reconnect leaks a full set of server processes.
        """
        if self._loop is None or self._shutdown is None or self._thread is None:
            return
        self._loop.call_soon_threadsafe(self._shutdown.set)
        self._thread.join(timeout=15)
        self._ready.clear()

    def _run_loop(self) -> None:
        """Thread entry point. Owns the event loop for the host's lifetime."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._serve())
        except BaseException as exc:  # surfaced to start() via _boot_error
            self._boot_error = exc
            self._ready.set()
        finally:
            loop.close()

    async def _serve(self) -> None:
        """Connect every server, then hold them all open until stop() is called."""
        self._shutdown = asyncio.Event()
        namer = Namer()

        async with ClientSessionGroup(component_name_hook=namer) as group:
            self._group = group
            for spec in self.specs:
                namer.current = spec.key
                self.status.append(await self._connect(group, spec))

            self._ready.set()
            await self._shutdown.wait()

        self._group = None

    async def _connect(self, group: ClientSessionGroup, spec: ServerSpec) -> ServerStatus:
        """Connect one server and report what it contributed. One failure is survivable."""
        before = set(group.tools)
        try:
            await group.connect_to_server(spec.params)
        except Exception as exc:
            return ServerStatus(spec.key, connected=False, error=str(exc))

        gained = sorted(set(group.tools) - before)
        return ServerStatus(spec.key, True, tools=gained, exposed=apply_allowlist(gained, spec))

    # ----- calling ---------------------------------------------------------

    def run(self, coro: Awaitable[T], timeout: float | None = None) -> T:
        """
        Run a coroutine on the host loop from a synchronous caller.

        Every Groq call and every MCP tool call goes through here, so the codebase
        has exactly one thread boundary.
        """
        if self._loop is None:
            raise RuntimeError("host is not running — call start() first")

        limit = timeout or config.TOOL_TIMEOUT
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=limit)
        except FutureTimeout:
            # Otherwise a slow fetch keeps running after the UI has given up on it.
            future.cancel()
            raise TimeoutError(f"timed out after {limit:.0f}s") from None

    def call_tool(self, name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        """Call a namespaced tool by its ``serverkey__toolname`` name."""
        if self._group is None:
            raise RuntimeError("host is not running")

        result = self.run(self._group.call_tool(name, arguments))
        if not isinstance(result, types.CallToolResult):
            # allow_input_required defaults to False, so this should not happen.
            raise RuntimeError(f"{name} asked for interactive input, which this host cannot do")
        return result

    # ----- inspection ------------------------------------------------------

    @property
    def tools(self) -> dict[str, types.Tool]:
        """Every tool across every connected server, keyed by namespaced name."""
        return dict(self._group.tools) if self._group is not None else {}

    @property
    def exposed_names(self) -> list[str]:
        """The subset of tool names actually offered to the model."""
        return [name for status in self.status for name in status.exposed]
