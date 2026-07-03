"""The five graph nodes and the two routers.

Flow: planner drafts -> validator checks (no LLM) -> approval gate freezes for a
human -> execute (mock irreversible action) or cancel. interrupt() lives in the
approval gate; the revision limit is enforced in both routers.
"""

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

from src.audit import record_event
from src.config import LEDGER, MAX_REVISIONS, OUTPUT_DIR, make_llm
from src.prompts import PLANNER_SYSTEM, revision_block
from src.state import ApprovalState, PlannedAction

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# --------------------------------------------------------------------------- #
# 1. Planner — the agentic step that drafts the action.
# --------------------------------------------------------------------------- #
def planner_node(state: ApprovalState) -> dict:
    """Draft (or re-draft) the action with the LLM using structured output."""
    llm = make_llm().with_structured_output(PlannedAction)

    system = PLANNER_SYSTEM
    feedback = state.get("feedback", "")
    errors = state.get("validation_errors", [])
    if feedback or errors:
        system += revision_block(feedback, errors)

    planned: PlannedAction = llm.invoke(
        [SystemMessage(content=system), HumanMessage(content=state["request"])]
    )
    revision = state.get("revision", 0) + 1
    record_event(
        state["thread_id"], "drafted", "agent",
        {"revision": revision, "summary": planned.summary, "type": planned.action_type},
    )
    # Note: we deliberately do NOT clear feedback or validation_errors here.
    # The validator overwrites validation_errors on its next run, and each human
    # decision overwrites feedback. Leaving them in place means an edit instruction
    # survives an internal re-draft if that draft also fails validation.
    return {
        "action": planned.model_dump(),
        "revision": revision,
        "history": [f"planner: drafted v{revision} ({planned.action_type})"],
    }


# --------------------------------------------------------------------------- #
# 2. Validator — deterministic policy gate. No LLM: rules are auditable.
# --------------------------------------------------------------------------- #
def _validate(action: dict) -> list[str]:
    """Return a list of problems with the draft. Empty list means it passes."""
    errors: list[str] = []
    if not action.get("subject", "").strip():
        errors.append("subject is empty")
    if not action.get("body", "").strip():
        errors.append("body is empty")

    recipient = action.get("recipient", "").strip()
    if not recipient:
        errors.append("recipient is empty")
    elif action.get("action_type") == "email" and not _EMAIL_RE.match(recipient):
        errors.append(f"recipient '{recipient}' is not a valid email address")
    return errors


def validator_node(state: ApprovalState) -> dict:
    """Check the draft against per-type rules before any human is bothered."""
    errors = _validate(state.get("action", {}))
    valid = not errors
    event = "validated" if valid else "validation_failed"
    record_event(state["thread_id"], event, "agent", {"errors": errors})
    line = "validator: passed" if valid else f"validator: failed ({'; '.join(errors)})"
    return {"valid": valid, "validation_errors": errors, "history": [line]}


def route_after_validation(state: ApprovalState) -> str:
    """Valid -> human gate. Invalid -> re-draft until the revision limit, then cancel."""
    if state.get("valid"):
        return "approval"
    if state.get("revision", 0) < MAX_REVISIONS:
        return "planner"
    return "cancel"


# --------------------------------------------------------------------------- #
# 3. Approval gate — the star. interrupt() freezes the graph for a human.
# --------------------------------------------------------------------------- #
def approval_gate_node(state: ApprovalState) -> dict:
    """Pause the graph and wait for a human approve / edit / reject decision.

    Everything BEFORE interrupt() re-runs when the graph resumes, so we keep this
    side-effect-free here. The "awaiting_approval" audit entry is written by the
    orchestrator (main.py) at the moment of each freeze, exactly once.
    """
    # Execution stops here. The value passed to Command(resume=...) on the next
    # invoke becomes the return value of interrupt().
    payload = interrupt({"draft": state["action"], "revision": state.get("revision")})

    decision = payload.get("decision", "reject")
    feedback = payload.get("feedback", "")
    record_event(
        state["thread_id"], "decision", "human",
        {"decision": decision, "feedback": feedback},
    )
    return {
        "decision": decision,
        "feedback": feedback,
        "history": [f"human: {decision}" + (f" ({feedback})" if feedback else "")],
    }


def route_after_approval(state: ApprovalState) -> str:
    """approve -> execute, reject -> cancel, edit -> re-draft (until the limit)."""
    decision = state.get("decision")
    if decision == "approve":
        return "execute"
    if decision == "reject":
        return "cancel"
    # edit: loop back to the planner unless we've hit the revision ceiling
    if state.get("revision", 0) < MAX_REVISIONS:
        return "planner"
    return "cancel"


# --------------------------------------------------------------------------- #
# 4. Execute — the mock irreversible action.
# --------------------------------------------------------------------------- #
_ICONS = {"email": "📧 EMAIL SENT", "calendar": "📅 EVENT BOOKED", "ticket": "🎫 TICKET CREATED"}


def _append_ledger(action: dict, thread_id: str) -> None:
    """Append the executed action to output/ledger.json (the mock 'send')."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ledger = json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else []
    ledger.append({"thread_id": thread_id, **action})
    LEDGER.write_text(json.dumps(ledger, indent=2), encoding="utf-8")


def execute_node(state: ApprovalState) -> dict:
    """Carry out the approved action (simulated) and log it to the ledger."""
    action = state["action"]
    _append_ledger(action, state["thread_id"])
    label = _ICONS.get(action.get("action_type"), "✅ ACTION DONE")
    result = f"{label} -> {action['recipient']} | {action['subject']}"
    print(result)
    record_event(state["thread_id"], "executed", "agent", {"summary": action.get("summary")})
    return {"status": "sent", "result": result, "history": [f"execute: {result}"]}


# --------------------------------------------------------------------------- #
# 5. Cancel — terminal path for reject or revision-limit exhaustion.
# --------------------------------------------------------------------------- #
def cancel_node(state: ApprovalState) -> dict:
    """End the run without acting. Reason is the rejection note or the limit."""
    if state.get("decision") == "reject":
        reason = state.get("feedback") or "rejected by reviewer"
    else:
        reason = f"revision limit reached ({MAX_REVISIONS})"
    result = f"Action cancelled: {reason}"
    print(result)
    record_event(state["thread_id"], "cancelled", "agent", {"reason": reason})
    return {"status": "cancelled", "result": result, "history": [f"cancel: {reason}"]}
