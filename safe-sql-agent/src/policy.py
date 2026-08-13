"""The access policy: exactly which columns and functions the agent may touch.

ALLOWED_COLUMNS is the single source of truth. Table access, the schema the
model is shown, the guard's rejection messages and the SQLite authorizer are
all derived from it, so there is one place to edit and nothing to keep in sync.

Everything here is default-deny: a table not listed does not exist as far as
the agent is concerned, and a column not listed cannot be read.

Inputs: the live database file (for column types and for spotting what was
deliberately left out).
Outputs: the allowlists, and the schema string that goes into the prompt.
"""

import sqlite3
from functools import lru_cache

from src.config import DB_PATH

# The whole policy. Note what is missing: customers.password_hash, and the
# payment_methods table entirely.
ALLOWED_COLUMNS: dict[str, frozenset[str]] = {
    "customers": frozenset({"id", "name", "email", "city", "country", "signup_date"}),
    "products": frozenset({"id", "name", "category", "price", "stock", "description"}),
    "orders": frozenset({"id", "customer_id", "order_date", "status", "total"}),
    "order_items": frozenset({"id", "order_id", "product_id", "quantity", "unit_price"}),
}

ALLOWED_TABLES: frozenset[str] = frozenset(ALLOWED_COLUMNS)

# SQL functions the agent may call, spelled the way SQLite reports them to the
# authorizer. Default-deny: anything off this list never runs, which is what
# stops load_extension() and readfile().
#
# Spelling matters more than it looks. The guard rebuilds SQL from the parse
# tree, and that rewrite renames some functions on the way out — substr()
# leaves as SUBSTRING(), ifnull() leaves as COALESCE(). What must be on this
# list is the name SQLite ends up seeing, not the one the model typed.
ALLOWED_FUNCTIONS: frozenset[str] = frozenset(
    {
        "count", "sum", "avg", "min", "max", "total",
        "round", "abs", "coalesce", "ifnull", "nullif",
        "length", "lower", "upper", "trim", "ltrim", "rtrim",
        "substr", "substring", "replace", "instr", "printf", "like", "glob",
        "date", "time", "datetime", "strftime", "julianday",
        "group_concat", "iif",
        # window functions — ordinary analytics, and SQLite asks about them too
        "row_number", "rank", "dense_rank", "ntile",
        "lag", "lead", "first_value", "last_value", "nth_value",
    }
)


def is_readable(table: str, column: str) -> bool:
    """Whether one resolved table.column pair may be read. Used by the authorizer."""
    return column.lower() in ALLOWED_COLUMNS.get(table.lower(), frozenset())


@lru_cache(maxsize=1)
def _real_schema() -> dict[str, list[tuple[str, str]]]:
    """Read the actual tables and columns out of the database, once."""
    schema: dict[str, list[tuple[str, str]]] = {}
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as conn:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'")]
        for table in tables:
            info = conn.execute(f"PRAGMA table_info({table})").fetchall()
            schema[table.lower()] = [(row[1], row[2]) for row in info]
    return schema


@lru_cache(maxsize=1)
def denied_tables() -> frozenset[str]:
    """Real tables the agent may not read.

    The guard uses this to stop a CTE borrowing the name of an off-limits
    table, which would otherwise let the query text widen the allowlist.
    """
    return frozenset(set(_real_schema()) - ALLOWED_TABLES)


@lru_cache(maxsize=1)
def denied_columns() -> frozenset[str]:
    """Columns that exist in allowed tables but were deliberately left out.

    Derived, never hand-written. The guard uses it to reject a query early with
    a reason the model can act on, rather than letting SQLite refuse it later
    with a message about permissions.
    """
    denied: set[str] = set()
    for table, columns in _real_schema().items():
        if table in ALLOWED_TABLES:
            allowed = ALLOWED_COLUMNS[table]
            denied |= {name.lower() for name, _ in columns if name.lower() not in allowed}
    return frozenset(denied)


def describe_schema(full: bool = False) -> str:
    """Render a schema description.

    By default this is what the model is shown — allowlisted columns only, so
    denied names never reach the prompt. full=True gives the real schema
    including the secrets, used only to simulate an unguarded agent.
    """
    lines: list[str] = []
    for table, columns in sorted(_real_schema().items()):
        if not full and table not in ALLOWED_TABLES:
            continue
        visible = [
            f"{name} {ctype}" for name, ctype in columns
            if full or name.lower() in ALLOWED_COLUMNS[table]
        ]
        lines.append(f"{table}({', '.join(visible)})")
    return "\n".join(lines)


def starter_policy() -> str:
    """Print an ALLOWED_COLUMNS block for whatever database is configured.

    Pointing this project at your own database means writing that dict. This
    generates it with everything included, so you can delete what should not
    be readable rather than type out what should.

    Run: PYTHONPATH=. uv run python -m src.policy
    """
    lines = ["ALLOWED_COLUMNS: dict[str, frozenset[str]] = {"]
    for table, columns in sorted(_real_schema().items()):
        names = ", ".join(f'"{name}"' for name, _ in columns)
        lines.append(f'    "{table}": frozenset({{{names}}}),')
    lines.append("}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(starter_policy())
