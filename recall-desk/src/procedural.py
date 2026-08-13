"""Procedural memory — the triage playbook that shapes how the desk behaves.

This tier is different from the other two in three ways that all follow from one
fact: it is NEVER retrieved by similarity, it is always in the prompt.

- Written with index=False, so it never pollutes the two searched tiers.
- Capped at MAX_RULES, because an always-on tier is a fixed token tax on every
  single turn. An uncapped playbook grows until it crowds out everything else.
- Holds BEHAVIOUR only, never diagnosis. Diagnosis belongs in episodic memory;
  putting it here would mean paying for it on every turn whether or not it is
  relevant.

Writes are versioned and gated on explicit human feedback. An agent that rewrites
its own operating instructions from an unprompted model call drifts within a few
turns.
"""

from __future__ import annotations

from enum import Enum

from langgraph.store.sqlite import SqliteStore
from pydantic import BaseModel, Field

from src.config import MAX_RULES
from src.llm import ask
from src.store import HISTORY_NS, PLAYBOOK_KEY, PLAYBOOK_NS

DEFAULT_RULES: list[str] = [
    "Ask for the request ID before proposing a fix, unless the customer already gave one.",
    "Reply in 5 lines or fewer.",
    "Never open with an apology or a preamble. Lead with the substance.",
    "Escalate any settlement delay over 24 hours. Never promise a resolution time or date.",
]


class Op(str, Enum):
    """What a feedback turn does to the playbook."""

    add = "add"
    replace = "replace"
    delete = "delete"
    none = "none"


class RuleEdit(BaseModel):
    """A single proposed change to the playbook.

    Every field is required — `gpt-oss-120b` enforces strict JSON schemas, so
    unused fields carry sentinels ("" and 0) rather than being omitted.
    """

    op: Op = Field(description="add, replace, delete, or none if no rule change is warranted")
    rule: str = Field(description="the new rule text; empty string for delete or none")
    target: int = Field(
        description="1-based index of the rule being replaced or deleted; 0 for add or none"
    )
    reason: str = Field(description="one short sentence on why")


class Playbook(BaseModel):
    """The playbook as stored."""

    version: int
    rules: list[str]

    def render(self) -> str:
        """Format for the prompt."""
        numbered = "\n".join(f"{i}. {r}" for i, r in enumerate(self.rules, 1))
        return f"How this desk operates (playbook v{self.version}):\n{numbered}"


def load(store: SqliteStore) -> Playbook:
    """Read the current playbook, seeding the default if none exists."""
    item = store.get(PLAYBOOK_NS, PLAYBOOK_KEY)
    if item is None:
        return _save(store, Playbook(version=1, rules=list(DEFAULT_RULES)))
    return Playbook(**item.value)


def _save(store: SqliteStore, playbook: Playbook) -> Playbook:
    """Write the playbook as current and snapshot it into history.

    index=False on both — this tier is never searched.
    """
    payload = playbook.model_dump()
    store.put(PLAYBOOK_NS, PLAYBOOK_KEY, payload, index=False)
    store.put(HISTORY_NS, f"v{playbook.version:03d}", payload, index=False)
    return playbook


def history(store: SqliteStore) -> list[Playbook]:
    """Every past version, oldest first."""
    items = store.search(HISTORY_NS, limit=100)
    books = [Playbook(**i.value) for i in items]
    return sorted(books, key=lambda b: b.version)


def apply_edit(store: SqliteStore, edit: RuleEdit) -> tuple[Playbook, str]:
    """Apply a proposed edit, enforcing the cap in code.

    Args:
        store: the memory store.
        edit: the proposed change.

    Returns:
        The playbook now in force, and a human-readable note about what happened.
        A rejected edit returns the unchanged playbook and the reason.
    """
    current = load(store)
    rules = list(current.rules)

    if edit.op is Op.none:
        return current, "no change proposed"

    if edit.op is Op.add:
        if len(rules) >= MAX_RULES:
            return current, (
                f"rejected: playbook is at the {MAX_RULES}-rule cap. "
                "Replace a rule instead of adding one."
            )
        rules.append(edit.rule)
        note = f"added rule {len(rules)}"

    elif edit.op is Op.replace:
        if not 1 <= edit.target <= len(rules):
            return current, f"rejected: no rule {edit.target} to replace"
        note = f"replaced rule {edit.target}: {rules[edit.target - 1]!r}"
        rules[edit.target - 1] = edit.rule

    else:  # delete
        if not 1 <= edit.target <= len(rules):
            return current, f"rejected: no rule {edit.target} to delete"
        note = f"deleted rule {edit.target}: {rules[edit.target - 1]!r}"
        rules.pop(edit.target - 1)

    updated = _save(store, Playbook(version=current.version + 1, rules=rules))
    return updated, f"v{current.version} -> v{updated.version}: {note}"


def rollback(store: SqliteStore, version: int) -> tuple[Playbook, str]:
    """Restore an earlier version's rules as a new version.

    History stays append-only: rolling back v4 to v2 creates v5 holding v2's
    rules, rather than deleting v3 and v4.

    Args:
        store: the memory store.
        version: which past version's rules to restore.

    Returns:
        The playbook now in force and a note.
    """
    target = next((b for b in history(store) if b.version == version), None)
    if target is None:
        return load(store), f"rejected: no v{version} in history"
    current = load(store)
    restored = _save(store, Playbook(version=current.version + 1, rules=list(target.rules)))
    return restored, f"v{current.version} -> v{restored.version}, restoring v{version}'s rules"


def propose_rule_edit(feedback: str, playbook: Playbook) -> RuleEdit:
    """Turn a supervisor's feedback into one playbook edit.

    Args:
        feedback: what the human said the desk should do differently.
        playbook: the playbook in force.

    Returns:
        The proposed edit. `Op.none` is a valid outcome.
    """
    at_cap = len(playbook.rules) >= MAX_RULES
    system = f"""You maintain the operating playbook for a payments API support desk.

A supervisor has given feedback. Turn it into exactly ONE edit.

The playbook governs BEHAVIOUR — tone, length, what to ask for, when to escalate.
It must never contain diagnostic knowledge about specific bugs; that belongs in
the ticket history instead.

Current playbook:
{playbook.render()}

{"The playbook is AT ITS CAP. You must use replace or delete, never add."
 if at_cap else f"The playbook holds {len(playbook.rules)} of {MAX_RULES} rules."}

If the feedback does not warrant a rule change, return op=none."""
    return ask(RuleEdit, system, feedback)


if __name__ == "__main__":
    # Cap, replace, and rollback — all without an API key.
    from src.store import open_store

    store = open_store(":memory:")
    book = load(store)
    print(book.render())

    print("\n-- fill to the cap --")
    for i in range(MAX_RULES):
        book, note = apply_edit(store, RuleEdit(op=Op.add, rule=f"Filler rule {i}.", target=0, reason="test"))
        print(f"  {note}")

    assert len(book.rules) == MAX_RULES, f"cap breached: {len(book.rules)} rules"
    print(f"\nheld at {len(book.rules)} rules (cap {MAX_RULES})")

    print("\n-- replace still works at the cap --")
    book, note = apply_edit(store, RuleEdit(op=Op.replace, rule="Never share another customer's data.", target=6, reason="test"))
    print(f"  {note}")
    assert book.rules[5].startswith("Never share"), "replace did not land"

    print("\n-- rollback to v1 --")
    book, note = apply_edit(store, RuleEdit(op=Op.delete, rule="", target=1, reason="test"))
    print(f"  {note}")
    book, note = rollback(store, 1)
    print(f"  {note}")
    assert book.rules == DEFAULT_RULES, "rollback did not restore v1's rules"
    print(f"\nversions on file: {[b.version for b in history(store)]}")
    print("\nOK — cap enforced, replace works at the cap, rollback restores.")
