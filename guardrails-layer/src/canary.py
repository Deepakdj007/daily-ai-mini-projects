"""Layer 5: has the system prompt leaked?

Inputs:  the assistant's answer
Outputs: whether the planted secret appears in it

The cheapest check in the stack and the only one with no false positives. A
string that exists in exactly one place - the system prompt - cannot appear in
an answer by coincidence. No model, no threshold, no judgement call.
"""

from __future__ import annotations

from src.config import CANARY


def _squash(text: str) -> str:
    """Reduce text to lowercase alphanumerics.

    A model asked to leak a secret it has been told not to leak will often
    volunteer it with spaces, dashes or newlines inserted. Squashing both
    sides catches that without a fuzzy matcher.
    """
    return "".join(ch for ch in text.lower() if ch.isalnum())


def leaked(text: str) -> bool:
    """True if the canary, or a recognisable chunk of it, is in the text."""
    haystack = _squash(text)
    needle = _squash(CANARY)
    if needle in haystack:
        return True
    # The distinctive middle of the token is enough. A model that emits
    # "7Q4X-ESCALATE" has leaked the secret even without the prefix.
    return _squash("7Q4X-ESCALATE") in haystack
