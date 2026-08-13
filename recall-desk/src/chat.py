"""Terminal REPL for the desk, showing every recall and every write.

The point of the display is that memory is never implicit. Each turn prints what
was pulled from which tier and at what similarity, and each write says which tier
it landed in.

Commands:
    /who <acme|beta>   switch customer (semantic memory is per customer)
    /memory            dump all three tiers
    /why               what was injected into the last turn, with scores
    /resolve           close the ticket and file an episode (episodic write)
    /feedback <text>   supervisor feedback -> a playbook edit (procedural write)
    /playbook          current rules plus version history
    /rollback <n>      restore version n's rules
    /new               start a fresh thread (drops conversation, keeps memory)
    /quit
"""

from __future__ import annotations

import uuid

from langchain_core.messages import AIMessage, HumanMessage
from rich.console import Console
from rich.panel import Panel

from src import episodic, procedural, semantic
from src.agent import ask_desk, build_graph, open_checkpointer
from src.config import CUSTOMERS
from src.store import open_store

console = Console()


def _show_trace(trace: dict) -> None:
    """Print what memory contributed to a turn."""
    facts, fscores = trace.get("facts", []), trace.get("fact_scores", [])
    eps, escores = trace.get("episodes", []), trace.get("episode_scores", [])
    rules = trace.get("rules", [])

    lines = []
    if facts:
        lines.append("[cyan]semantic[/cyan]")
        lines += [f"  {s:.3f}  {t}" for t, s in zip(facts, fscores)]
    if eps:
        lines.append("[magenta]episodic[/magenta]")
        lines += [f"  {s:.3f}  {t}" for t, s in zip(eps, escores)]
    if rules:
        lines.append(f"[yellow]procedural[/yellow]  {len(rules)} rules, always injected")
    if not lines:
        lines.append("[dim]nothing recalled[/dim]")

    tokens = trace.get("tokens", {})
    footer = (
        f"{trace.get('summary', '')}\n"
        f"prompt {trace.get('system_chars', 0)} chars -> "
        f"{tokens.get('input', 0)} input tokens, {tokens.get('output', 0)} output"
    )
    if trace.get("written_facts"):
        footer += f"\n[green]wrote semantic:[/green] {', '.join(trace['written_facts'])}"
    console.print(Panel("\n".join(lines) + "\n\n[dim]" + footer + "[/dim]", title="memory", expand=False))


def _dump_memory(store, customer_id: str) -> None:
    """Show all three tiers at once."""
    console.print(f"\n[bold cyan]semantic[/bold cyan] — {CUSTOMERS[customer_id]} only")
    for item in semantic.all_facts(store, customer_id):
        console.print(f"  [{item.key}] {item.value['text']}")

    console.print("\n[bold magenta]episodic[/bold magenta] — global, every customer")
    for item in episodic.all_episodes(store):
        console.print(f"  {item.value['text']}")
        console.print(f"    [dim]cause: {item.value['root_cause']}[/dim]")

    book = procedural.load(store)
    console.print(f"\n[bold yellow]procedural[/bold yellow] — global\n{book.render()}")


def _resolve(store, thread_messages: list) -> None:
    """File an episode from the conversation so far."""
    if len(thread_messages) < 2:
        console.print("[yellow]nothing to summarise yet[/yellow]")
        return
    transcript = "\n".join(
        f"{'Customer' if isinstance(m, HumanMessage) else 'Agent'}: {m.content}"
        for m in thread_messages
        if isinstance(m, (HumanMessage, AIMessage))
    )
    console.print("[dim]summarising the ticket...[/dim]")
    episode = episodic.summarize_episode(transcript)
    if episode is None:
        console.print("[yellow]not worth keeping — nothing was actually resolved[/yellow]")
        return
    key = episodic.append(store, episode)
    console.print(
        Panel(
            f"[bold]{episode.text}[/bold]\n"
            f"ruled out : {', '.join(episode.tried)}\n"
            f"cause     : {episode.root_cause}\n"
            f"fix       : {episode.fix}",
            title=f"episode filed ({key}) — scrubbed and global",
            expand=False,
        )
    )


def _feedback(store, text: str) -> None:
    """Turn supervisor feedback into a playbook edit."""
    if not text.strip():
        console.print("[yellow]usage: /feedback stop promising settlement dates[/yellow]")
        return
    book = procedural.load(store)
    console.print("[dim]proposing a playbook edit...[/dim]")
    edit = procedural.propose_rule_edit(text, book)
    updated, note = procedural.apply_edit(store, edit)
    console.print(f"[green]{note}[/green]  [dim]({edit.reason})[/dim]")
    console.print(updated.render())


def main() -> None:
    store = open_store()
    graph = build_graph(store, open_checkpointer())
    customer_id = "acme"
    thread_id = f"chat-{uuid.uuid4().hex[:8]}"
    last_trace: dict = {}

    console.print(
        Panel(
            "PaySetu support desk. Three memory tiers, all visible.\n"
            "[dim]/memory /why /resolve /feedback <text> /playbook /rollback <n> "
            "/who <acme|beta> /new /quit[/dim]",
            title="recall-desk",
            expand=False,
        )
    )
    console.print(f"[dim]customer: {CUSTOMERS[customer_id]}  thread: {thread_id}[/dim]")

    while True:
        try:
            line = console.input(f"\n[bold]{customer_id}>[/bold] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue

        if line in ("/quit", "/exit"):
            break
        if line == "/memory":
            _dump_memory(store, customer_id)
            continue
        if line == "/why":
            if last_trace:
                _show_trace(last_trace)
            else:
                console.print("[yellow]no turn yet[/yellow]")
            continue
        if line == "/playbook":
            console.print(procedural.load(store).render())
            versions = [b.version for b in procedural.history(store)]
            console.print(f"[dim]versions on file: {versions}[/dim]")
            continue
        if line.startswith("/rollback"):
            parts = line.split()
            if len(parts) != 2 or not parts[1].isdigit():
                console.print("[yellow]usage: /rollback 1[/yellow]")
                continue
            book, note = procedural.rollback(store, int(parts[1]))
            console.print(f"[green]{note}[/green]\n{book.render()}")
            continue
        if line.startswith("/who"):
            parts = line.split()
            if len(parts) != 2 or parts[1] not in CUSTOMERS:
                console.print(f"[yellow]usage: /who {'|'.join(CUSTOMERS)}[/yellow]")
                continue
            customer_id = parts[1]
            thread_id = f"chat-{uuid.uuid4().hex[:8]}"
            console.print(
                f"[green]now {CUSTOMERS[customer_id]}[/green] — new thread, "
                "different semantic memory, same episodes and playbook"
            )
            continue
        if line == "/new":
            thread_id = f"chat-{uuid.uuid4().hex[:8]}"
            console.print(f"[green]new thread {thread_id}[/green] — memory is untouched")
            continue
        if line == "/resolve":
            state = graph.get_state({"configurable": {"thread_id": thread_id}})
            _resolve(store, (state.values or {}).get("messages", []))
            continue
        if line.startswith("/feedback"):
            _feedback(store, line[len("/feedback"):])
            continue
        if line.startswith("/"):
            console.print(f"[yellow]unknown command {line.split()[0]}[/yellow]")
            continue

        state = ask_desk(graph, customer_id, line, thread_id=thread_id)
        last_trace = state.get("trace", {})
        _show_trace(last_trace)
        console.print(
            Panel(
                state["reply"],
                title=f"reply  [severity={state['severity']} escalate={state['escalate']}]",
                expand=False,
            )
        )

    console.print("\n[dim]memory persists. Run again and it remembers.[/dim]")


if __name__ == "__main__":
    main()
