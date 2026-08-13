"""The guard: parse the model's SQL, judge it against the policy, rewrite it.

Nothing here talks to an LLM and nothing here trusts a string. The model's SQL
is turned into a syntax tree, every node is checked, and the SQL that finally
runs is printed back out of the tree — so anything the checks did not see
cannot survive into execution.

Inputs: one SQL string from the model.
Outputs: a Verdict — either safe rewritten SQL, or a blocked reason to feed back.
"""

from dataclasses import dataclass

import sqlglot
from sqlglot import exp

from src.config import MAX_ROWS
from src.policy import ALLOWED_TABLES, denied_columns, denied_tables

# Statement types that must never appear anywhere in the tree — not at the top,
# not inside a subquery. exp.Command is sqlglot's catch-all for statements it
# does not model (VACUUM, and anything exotic), so denying it is deliberate.
FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter,
    exp.Attach, exp.Detach, exp.Pragma, exp.Set, exp.Use, exp.Grant,
    exp.Transaction, exp.Commit, exp.Rollback, exp.Copy, exp.Merge,
    exp.TruncateTable, exp.Command,
)


@dataclass(frozen=True)
class Verdict:
    """Result of guarding one SQL string.

    ok: whether the query may run.
    sql: the rewritten SQL to execute (only meaningful when ok is True).
    reason: plain-English rejection sent back to the model when ok is False.
    layer: which check decided, for the trace shown in the UI.
    """

    ok: bool
    sql: str = ""
    reason: str = ""
    layer: str = ""


def _block(layer: str, reason: str) -> Verdict:
    """Build a rejection verdict."""
    return Verdict(ok=False, reason=reason, layer=layer)


def _check_tables(tree: exp.Expression) -> str | None:
    """Every table referenced must be allowlisted or a CTE defined in this query.

    A CTE may not borrow the name of a table that is off limits: allowing it
    would let the query text widen the allowlist, and reasoning about which
    name won would be one step more thinking than a safety check deserves.
    """
    allowed_list = ", ".join(sorted(ALLOWED_TABLES))
    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    shadowed = sorted(cte_names & denied_tables())
    if shadowed:
        return f"a CTE may not reuse the name of a table you cannot read ('{shadowed[0]}')."

    for table in tree.find_all(exp.Table):
        # main.customers is fine; anything else is a second database.
        if table.db and table.db.lower() != "main":
            return f"database '{table.db}' is not reachable."
        if table.catalog:
            return f"catalog '{table.catalog}' is not reachable."
        name = table.name.lower()
        if not name:
            return "table-valued functions are not allowed."
        if name not in ALLOWED_TABLES and name not in cte_names:
            return f"table '{table.name}' is not readable. Allowed tables: {allowed_list}."
    return None


def _check_projections(tree: exp.Expression) -> str | None:
    """Reject SELECT * and t.* — a star would smuggle past the column check.

    COUNT(*) is fine: the star lives inside a function, not in the select list.
    """
    for select in tree.find_all(exp.Select):
        for item in select.expressions:
            starred = isinstance(item, exp.Star) or (
                isinstance(item, exp.Column) and isinstance(item.this, exp.Star)
            )
            if starred:
                return "SELECT * is not allowed. Name the columns you need."
    return None


def _check_columns(tree: exp.Expression) -> str | None:
    """Reject any reference to a column the policy left out, wherever it appears.

    This is a denylist on purpose. The guard sees names before SQLite resolves
    them, so it cannot tell the column `revenue` from an alias named `revenue`
    in ORDER BY — an allowlist here would reject ordinary queries. The exact
    table.column allowlist lives in the authorizer, where names are resolved.
    """
    denied = denied_columns()
    for column in tree.find_all(exp.Column):
        if column.name.lower() in denied:
            return f"column '{column.name}' is confidential and cannot be read."
    return None


def _check_functions(tree: exp.Expression) -> str | None:
    """Reject functions sqlglot does not recognise.

    An unknown name parses as exp.Anonymous — that is where load_extension()
    and readfile() land. Known functions are ordinary SQL and pass here; the
    SQLite authorizer checks them again by name at execution time.
    """
    for func in tree.find_all(exp.Anonymous):
        return f"function '{func.name}()' is not on the allowed list."
    return None


def _clamp_limit(tree: exp.Expression) -> exp.Expression:
    """Force a LIMIT no larger than MAX_ROWS, keeping a smaller one if present.

    Only a bare non-negative integer counts as an existing limit. Reading the
    number out of anything else is how this check gets fooled: LIMIT 40 + 60
    parses as an Add whose name is "40", and LIMIT -1 as a Neg whose name is
    "1" — and in SQLite, LIMIT -1 means no limit at all.
    """
    current = tree.args.get("limit")
    if current is not None:
        value = current.expression
        if isinstance(value, exp.Literal) and not value.is_string and value.name.isdigit():
            if int(value.name) <= MAX_ROWS:
                return tree
    return tree.limit(MAX_ROWS)


def guard(raw_sql: str) -> Verdict:
    """Run every static check over one SQL string and return the safe rewrite."""
    text = raw_sql.strip().strip(";").strip()
    if not text:
        return _block("syntax", "empty query.")

    try:
        statements = sqlglot.parse(text, read="sqlite")
    except sqlglot.ParseError as err:
        return _block("syntax", f"not valid SQLite: {str(err).splitlines()[0]}")

    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        return _block("single statement", f"send exactly one statement, got {len(statements)}.")

    # A UNION / EXCEPT / INTERSECT of SELECTs is still a read, and every check
    # below walks the whole tree, so both branches get the same treatment.
    tree = statements[0]
    if not isinstance(tree, (exp.Select, exp.SetOperation)):
        return _block("read only", f"only SELECT is allowed, got {type(tree).__name__.upper()}.")
    for node in tree.walk():
        if isinstance(node, FORBIDDEN_NODES):
            return _block("read only", f"{type(node).__name__.upper()} is not allowed.")
    if any(tree.find_all(exp.Into)):
        return _block("read only", "SELECT ... INTO writes to the database and is not allowed.")

    for layer, check in (
        ("table allowlist", _check_tables),
        ("no star", _check_projections),
        ("column allowlist", _check_columns),
        ("function allowlist", _check_functions),
    ):
        problem = check(tree)
        if problem:
            return _block(layer, problem)

    safe = _clamp_limit(tree.copy())
    for node in safe.walk():
        node.comments = None
    return Verdict(ok=True, sql=safe.sql(dialect="sqlite"), layer="passed")
