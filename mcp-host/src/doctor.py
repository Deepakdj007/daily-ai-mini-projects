"""
Connection check for the MCP host, without Streamlit in the way.

Boots every configured server, prints what connected and what each one offers,
then shuts down cleanly. When something breaks, run this first: it tells you
whether the problem is MCP or the UI.

Inputs: servers.json.
Outputs: a table on stdout; exit code 1 if any server failed.
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.table import Table

from src.host import MCPHost
from src.inventory import NAME_SEP


def main() -> int:
    """Boot the host, report every server, and return a shell exit code."""
    console = Console()
    host = MCPHost()

    console.print(f"[dim]booting {len(host.specs)} MCP servers (first run downloads them)...[/dim]")
    try:
        host.start()
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1

    try:
        table = Table(title="MCP servers", header_style="bold")
        table.add_column("server")
        table.add_column("status")
        table.add_column("exposed", justify="right")
        table.add_column("tools")

        for status in host.status:
            if status.connected:
                names = ", ".join(n.split(NAME_SEP, 1)[1] for n in status.exposed)
                table.add_row(
                    status.key,
                    "[green]ok[/green]",
                    f"{len(status.exposed)}/{len(status.tools)}",
                    names or "[dim]none exposed[/dim]",
                )
            else:
                table.add_row(status.key, "[red]failed[/red]", "-", f"[red]{status.error}[/red]")

        console.print(table)

        connected = sum(1 for s in host.status if s.connected)
        console.print(
            f"\n[bold]{connected}/{len(host.status)} servers connected[/bold] | "
            f"{len(host.tools)} tools discovered | {len(host.exposed_names)} sent to the model"
        )
        return 0 if connected == len(host.status) else 1
    finally:
        host.stop()


if __name__ == "__main__":
    sys.exit(main())
