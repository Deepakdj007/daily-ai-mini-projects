"""Turning comparison rows into a table, and flagging the ones that prove nothing.

Kept apart from the running of the experiment, so the rules about what makes a
row trustworthy live in one readable place.

Inputs:  the row dicts produced by src/compare.run_one
Outputs: a rich Table, a list of warnings, and output/compare.md
"""

from pathlib import Path

from rich.table import Table

from src import config

COLUMNS: tuple[str, ...] = (
    "brief", "kind", "policy", "who pulled the cord", "turns", "questions",
    "answer in report", "researchers ok", "sec",
)


def who(row: dict) -> str:
    """Name who pulled the cord, or who wanted to and was not allowed."""
    if row["asked"]:
        return ", ".join(dict.fromkeys(row["asked"])).replace("research ", "")
    if row["wanted"]:
        silenced = ", ".join(dict.fromkeys(row["wanted"])).replace("research ", "")
        return f"(silenced: {silenced})"
    return "nobody"


def score(row: dict) -> str:
    """Planted tokens recovered, or a dash when the brief planted none."""
    if row["tokens_total"] == 0:
        return "-"
    return f"{row['tokens_found']}/{row['tokens_total']}"


def researchers_ok(row: dict) -> str:
    """How many researchers actually answered, out of how many ran.

    A row where a researcher crashed says nothing about when agents ask, so this
    has to be visible in the table and not buried in a warning.
    """
    total = config.NUM_RESEARCHERS
    return f"{total - len(row['failures'])}/{total}"


def _cells(row: dict) -> list[str]:
    """One row of the table, in column order."""
    return [
        row["id"], row["kind"], row["policy"], who(row), str(row["turns"]),
        str(row["questions"]), score(row), researchers_ok(row), str(row["seconds"]),
    ]


def render(rows: list[dict]) -> Table:
    """Build the comparison table."""
    table = Table(title="Same prompts. The only difference is who may ask.")
    for column in COLUMNS:
        table.add_column(column)
    for row in rows:
        table.add_row(*_cells(row))
    return table


def warnings(rows: list[dict]) -> list[str]:
    """Flag anything that would make the table dishonest.

    Print these, never suppress them. A planted token found with nobody to ask
    means the experiment leaked. A question on a complete brief is a false alarm.
    Either one belongs in the output as loudly as a success.
    """
    notes: list[str] = []
    for row in rows:
        if row["failures"]:
            notes.append(
                f"{row['id']} ({row['policy']}): {len(row['failures'])} researcher(s) "
                f"never answered - {row['failures'][0]}. This row says nothing "
                "about asking."
            )
        if row["dry"]:
            notes.append(
                f"{row['id']} ({row['policy']}): {row['dry']} researcher(s) got zero "
                "search results, so they had nothing to be stuck on."
            )
        if row["policy"] == "never" and row["tokens_found"]:
            notes.append(
                f"{row['id']}: a planted token reached the report with nobody to "
                "ask. It leaked from the brief or from a search result."
            )
        if row["kind"] == "clear" and row["turns"]:
            notes.append(
                f"{row['id']} ({row['policy']}): the cord was pulled on a brief "
                "that was already complete. That is a false alarm."
            )
    return notes


def write_markdown(rows: list[dict], notes: list[str], n_briefs: int) -> Path:
    """Save the same table to output/compare.md and return the path."""
    lines = [
        "# andon-cord comparison",
        "",
        f"Model `{config.MODEL}`, {config.NUM_RESEARCHERS} researchers, "
        f"{n_briefs} briefs. Prompts are identical in both policies.",
        "",
        "| " + " | ".join(COLUMNS) + " |",
        "|" + "---|" * len(COLUMNS),
    ]
    lines += ["| " + " | ".join(_cells(row)) + " |" for row in rows]
    if notes:
        lines += ["", "## Warnings", ""] + [f"- {n}" for n in notes]

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.OUTPUT_DIR / "compare.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
