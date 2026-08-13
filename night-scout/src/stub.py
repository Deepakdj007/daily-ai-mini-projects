"""A deterministic stand-in for the Groq call, for running the night with no key.

Inputs:  the same schema and messages call_structured() takes
Outputs: a canned instance of that schema

Why this exists: the interesting parts of an ambient agent are the clock, the memory
and the parked-draft handoff, and none of them need a real model to exercise. Stub
mode lets you watch a whole night, fill the inbox and click Approve without spending
a token — and it makes the behaviour reproducible, which a live model never is.
"""

import hashlib
import re

from src.state import Draft, TriageBatch, Verdict

_ITEM_ID = re.compile(r"item_id: (\S+)")
_TITLE = re.compile(r"^title: (.+)$", re.MULTILINE)


def _stable_score(item_id: str) -> int:
    """A repeatable 0-10 score derived from the id, so runs are comparable.

    Roughly a third of items land at 7 or above, which is enough to fill the
    carry-over queue and show the per-wake draft budget doing its job.
    """
    return hashlib.sha256(item_id.encode()).digest()[0] % 11


def _triage(prompt: str) -> TriageBatch:
    """Score every item_id found in the rendered list."""
    verdicts = [
        Verdict(
            item_id=item_id,
            score=_stable_score(item_id),
            reason="stub verdict, no model was called",
        )
        for item_id in _ITEM_ID.findall(prompt)
    ]
    return TriageBatch(verdicts=verdicts)


def _draft(prompt: str) -> Draft:
    """Echo the item's own title back as a plausible-looking entry."""
    found = _TITLE.search(prompt)
    title = found.group(1).strip() if found else "Untitled item"
    return Draft(
        headline=f"[stub] {title}"[:90],
        why_it_matters=(
            "This entry was generated without calling a model, so treat the wording "
            "as filler. The parking, approval and commit path around it is real."
        ),
        key_points=[
            "Stub mode: no Groq request was made for this draft",
            "Approve or reject it to exercise the cross-process resume",
        ],
        tags=["stub"],
    )


def respond(schema, messages):
    """Route to the right canned response for `schema`."""
    prompt = "\n".join(text for _role, text in messages)
    if schema is TriageBatch:
        return _triage(prompt)
    if schema is Draft:
        return _draft(prompt)
    raise NotImplementedError(f"stub mode has no response for {schema.__name__}")
