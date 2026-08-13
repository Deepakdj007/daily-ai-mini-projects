"""SqliteStore wiring and the namespace layout for the three memory tiers.

One SQLite file holds all three tiers, separated by namespace:

    ("facts", <customer_id>)   semantic  — per customer, indexed
    ("episodes",)              episodic  — global across customers, indexed
    ("playbook",)              procedural— global, NEVER indexed

Vector search is provided by sqlite-vec, loaded by the store itself when an
index config is present.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.store.sqlite import SqliteStore

from src.config import DIMS, STORE_PATH
from src.embed import embed_texts

# Only the "text" field of a stored value is embedded. Everything else in the
# value dict is carried along, returned on search, and never vectorised.
INDEX = {"dims": DIMS, "embed": embed_texts, "fields": ["text"]}

EPISODES_NS: tuple[str, ...] = ("episodes",)
PLAYBOOK_NS: tuple[str, ...] = ("playbook",)
PLAYBOOK_KEY = "current"
HISTORY_NS: tuple[str, ...] = ("playbook", "history")


def facts_ns(customer_id: str) -> tuple[str, ...]:
    """Namespace holding one customer's semantic facts."""
    return ("facts", customer_id)


def open_store(path: Path | str = STORE_PATH) -> SqliteStore:
    """Open the memory store and run migrations.

    Built by direct construction rather than `SqliteStore.from_conn_string`,
    which is a context manager that closes the connection on exit — the wrong
    shape for a long-lived REPL or a Streamlit `st.cache_resource` singleton.

    Args:
        path: SQLite file path, or ":memory:" for a throwaway store.

    Returns:
        A SqliteStore with vector search enabled and tables created.
    """
    conn = sqlite3.connect(
        str(path),
        check_same_thread=False,  # Streamlit and the graph touch this off-thread
        isolation_level=None,  # autocommit, matching from_conn_string
    )
    store = SqliteStore(conn, index=INDEX)
    store.setup()
    return store


def wipe(store: SqliteStore) -> None:
    """Delete every item in all three tiers. Used by seed.py for a clean run."""
    for namespace in store.list_namespaces():
        for item in store.search(namespace, limit=1000):
            store.delete(namespace, item.key)


if __name__ == "__main__":
    # Round-trip proof: write to two tiers, search one, print real scores.
    store = open_store(":memory:")

    store.put(("episodes",), "ep1", {"text": "Webhooks stopped firing overnight."})
    store.put(("episodes",), "ep2", {"text": "Customer was charged twice for one order."})
    store.put(PLAYBOOK_NS, PLAYBOOK_KEY, {"rules": ["be terse"]}, index=False)

    print("search 'duplicate payment taken from card':")
    for hit in store.search(("episodes",), query="duplicate payment taken from card"):
        print(f"  {hit.score:.4f}  {hit.key}  {hit.value['text']}")

    print("\nplaybook is stored but unindexed:")
    print(f"  get                 -> {store.get(PLAYBOOK_NS, PLAYBOOK_KEY).value}")
    print(f"  search(query=...)   -> {len(store.search(PLAYBOOK_NS, query='be terse'))} hits")
    print(f"  search(no query)    -> {len(store.search(PLAYBOOK_NS))} hits")
    print("\nAn index=False item is invisible to vector search — not ranked last,")
    print("absent. It is reachable only by get() or an unfiltered list.")
    print("\nOK")
