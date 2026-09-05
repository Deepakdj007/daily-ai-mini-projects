"""The guardrails layer itself: which checks run, in what order, and why.

Inputs:  a user turn, the retrieved documents, a GuardConfig
Outputs: an InputDecision before the model runs, an OutputDecision after

Each configuration below turns exactly one mechanism on top of the previous
one. That is what makes the leaderboard readable: a number that moves between
two rows moved because of one change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src import canary, pii, sanitize
from src.config import DOC_QUARANTINE_AT, USER_ESCALATE_AT
from src.corpus import Doc
from src.normalize import normalize
from src.policy import judge
from src.promptguard import scan


@dataclass(frozen=True)
class GuardConfig:
    """One rung of the ablation ladder."""

    name: str
    hardened_prompt: bool = False
    normalize_text: bool = False
    scan_user: bool = False
    escalate_to_judge: bool = False
    scan_docs: bool = False
    pii_firewall: bool = False
    canary_check: bool = False


@dataclass
class InputDecision:
    """What the input side of the layer concluded."""

    allowed: bool
    user_text: str
    docs: list[Doc]
    quarantined: list[str] = field(default_factory=list)
    blocked_by: str = ""
    reason: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass
class OutputDecision:
    """What the output side of the layer concluded."""

    text: str
    blocked_by: str = ""
    pii_found: list[tuple[str, str]] = field(default_factory=list)
    canary_leaked: bool = False
    reasoning_stripped: bool = False


REFUSAL = (
    "I can't help with that request. If you need account changes, raise a "
    "ticket from the app and a support agent will take it forward."
)

# Marks a turn where the model produced no answer at all, only truncated
# reasoning. Named so the scoring can tell it apart from a real block.
TRUNCATED = "no answer (truncated reasoning)"


async def check_input(
    user_text: str, docs: list[Doc], cfg: GuardConfig
) -> InputDecision:
    """Run the input-side layers in order, cheapest first."""
    notes: list[str] = []
    text = user_text

    if cfg.normalize_text:
        text, findings = normalize(text)
        notes.extend(findings)

    if cfg.scan_user:
        result = await scan(text)
        notes.append(f"prompt-guard user score {result.score:.3f}")
        if result.score >= USER_ESCALATE_AT:
            if cfg.escalate_to_judge:
                # The score is a trigger, not a verdict. Ordinary customers
                # write "ignore the previous quote" and mean it innocently.
                blocked, _ = await judge(text)
                notes.append(f"policy judge: {'BLOCK' if blocked else 'ALLOW'}")
                if blocked:
                    return InputDecision(
                        False, text, [], blocked_by="policy judge",
                        reason="policy violation", notes=notes,
                    )
            else:
                return InputDecision(
                    False, text, [], blocked_by="prompt-guard",
                    reason=f"score {result.score:.3f}", notes=notes,
                )

    return await _filter_docs(text, docs, cfg, notes)


async def _filter_docs(
    text: str, docs: list[Doc], cfg: GuardConfig, notes: list[str]
) -> InputDecision:
    """Normalize and scan each retrieved document, dropping poisoned ones."""
    kept: list[Doc] = []
    quarantined: list[str] = []

    for doc in docs:
        body = doc.body
        if cfg.normalize_text:
            body, findings = normalize(body)
            notes.extend(f"{doc.doc_id}: {f}" for f in findings)
        if cfg.scan_docs:
            result = await scan(body)
            if result.score >= DOC_QUARANTINE_AT:
                # A help article has no business addressing the assistant.
                # Dropping it costs a legitimate user nothing.
                quarantined.append(doc.doc_id)
                notes.append(f"{doc.doc_id}: quarantined at {result.score:.3f}")
                continue
        kept.append(Doc(doc.doc_id, doc.title, body, doc.poisoned))

    return InputDecision(True, text, kept, quarantined=quarantined, notes=notes)


def check_output(
    answer: str,
    cfg: GuardConfig,
    known_values: set[str],
    protected: dict[str, str] | None = None,
) -> OutputDecision:
    """Run the output-side layers. None of them consults a model."""
    if not (cfg.pii_firewall or cfg.canary_check):
        return OutputDecision(text=answer)

    # Reasoning first. Everything below scans whatever this returns, and a
    # model that refuses in its answer will still have quoted the record in
    # its chain of thought.
    text, had_reasoning = sanitize.strip_reasoning(answer)

    if had_reasoning and not text.strip():
        # The whole response was reasoning, usually a truncated one. There is
        # no answer to show, and showing the deliberation is what we stripped
        # it to avoid. This is a service failure, not a defence: the
        # evaluation reports it as "no answer" so it never scores as a guard
        # win, because a token-budget bug that reads as a perfect security
        # result is worse than no measurement at all.
        return OutputDecision(
            text=REFUSAL, blocked_by=TRUNCATED, reasoning_stripped=True,
        )

    if cfg.canary_check and canary.leaked(text):
        # A leaked system prompt is not something to redact around.
        return OutputDecision(
            text=REFUSAL, blocked_by="canary", canary_leaked=True,
            reasoning_stripped=had_reasoning,
        )

    found: list[tuple[str, str]] = []
    if cfg.pii_firewall:
        # Deny list before Presidio: these values are known, so there is no
        # reason to let a confidence score decide their fate.
        text, denied = pii.deny_list(text, protected or {})
        found.extend(denied)
        result = pii.firewall(text, known_values=known_values)
        text = result.text
        found.extend(result.found)

    return OutputDecision(
        text=text, pii_found=found, reasoning_stripped=had_reasoning
    )
