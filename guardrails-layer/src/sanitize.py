"""Strip model scaffolding before anything else reads the answer.

Inputs:  raw model output
Outputs: the part of it a customer should see

Some models put their chain of thought in message.content rather than in a
separate field. That text routinely quotes the data the model then decides not
to share - so an assistant can refuse in its answer and leak in its reasoning.
Anything downstream of here would be scanning the wrong string.
"""

from __future__ import annotations

import re

# Closed blocks first, then an unterminated opener, which is what a truncated
# response leaves behind.
THINK_BLOCK = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.I)
THINK_OPEN = re.compile(r"<(think|thinking|reasoning)>.*", re.DOTALL | re.I)


def strip_reasoning(text: str) -> tuple[str, bool]:
    """Remove reasoning blocks. Returns (clean text, whether any were found)."""
    cleaned = THINK_BLOCK.sub("", text)
    cleaned = THINK_OPEN.sub("", cleaned)
    stripped = cleaned != text
    return cleaned.strip(), stripped
