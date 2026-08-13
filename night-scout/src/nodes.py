"""The graph's nodes: read the item, draft an entry, park it, then act on the answer.

Inputs:  ScoutState
Outputs: partial ScoutState dicts

One rule governs this file. interrupt() does not resume mid-function — LangGraph
replays the whole node from the top — so nothing before the interrupt may write to
disk or the write happens twice.
"""

from datetime import datetime, timezone

from langgraph.types import interrupt

from src import config, sources
from src.llm import call_structured
from src.state import Draft, ScoutState

DRAFT_SYSTEM = """You write one reading-list entry for a developer's morning digest.

Their interest profile:
{profile}

Write for someone deciding in five seconds whether to open the link. No hype, no
"game-changing", no restating the title. If the item turns out to be thin, say so
plainly in why_it_matters rather than inflating it.

headline: a plain-English claim, not a title. Under 90 characters.
why_it_matters: two sentences, addressed as "you".
key_points: 2-4 bullets, each under 20 words, concrete and specific.
tags: 1-3 lowercase topic tags."""

REVISION = """
The reader saw your previous draft and asked for changes.

Your previous draft:
{previous}

What they asked for: {feedback}

Rewrite the entry accordingly. Change what they asked about and leave the rest."""


def detail_node(state: ScoutState) -> dict:
    """Get the best text available for this item. No LLM call.

    arXiv is skipped deliberately: the Atom summary already IS the abstract, so a
    page fetch would spend a request to get less than we have. Any failure falls
    back to the triage snippet, which is always enough to draft something honest.
    """
    item = state["item"]
    if item["source"] == "arxiv":
        return {"detail": item["snippet"], "history": ["detail: arxiv abstract"]}

    try:
        page = sources.fetch_page(item["url"])
    except Exception as exc:  # noqa: BLE001 — a bad page must not lose the item
        return {
            "detail": item["snippet"],
            "history": [f"detail: fetch failed ({type(exc).__name__}), using snippet"],
        }

    text = sources.clean(page, config.DETAIL_MAX_CHARS)
    if len(text) < 200:  # a paywall, a JS shell, or a bare redirect page
        return {"detail": item["snippet"], "history": ["detail: page too thin"]}
    return {"detail": text, "history": [f"detail: fetched {len(text)} chars"]}


def draft_node(state: ScoutState) -> dict:
    """Write the reading-list entry. The only LLM call inside the graph.

    Re-entered when the human asks for an edit, with their feedback appended.
    """
    item = state["item"]
    revision = state.get("revision", 0) + 1

    human = (
        f"source: {item['source']}\n"
        f"title: {item['title']}\n"
        f"url: {item['url']}\n"
        f"triage score: {state.get('score')} ({state.get('reason', '')})\n\n"
        f"content:\n{state.get('detail', item['snippet'])}"
    )
    if state.get("feedback"):
        human += REVISION.format(
            previous=state.get("draft", {}), feedback=state["feedback"]
        )

    draft = call_structured(
        Draft,
        [
            ("system", DRAFT_SYSTEM.format(profile=config.INTEREST_PROFILE)),
            ("human", human),
        ],
    )
    return {
        "draft": draft.model_dump(),
        "revision": revision,
        "history": [f"draft: revision {revision}"],
    }


def gate_node(state: ScoutState) -> dict:
    """Park the draft and wait for a human. May wait hours, in another process.

    Everything before interrupt() re-runs on resume, so nothing here touches disk.
    The 'parked' index row is written by the caller once, right after invoke()
    returns — the same trick approval-gate-agent uses for its audit log.
    """
    item = state["item"]
    payload = interrupt(
        {
            "item_id": item["item_id"],
            "source": item["source"],
            "title": item["title"],
            "url": item["url"],
            "score": state.get("score"),
            "reason": state.get("reason"),
            "draft": state["draft"],
            "revision": state.get("revision", 1),
        }
    )
    decision = payload.get("decision", "reject")
    return {
        "decision": decision,
        "feedback": payload.get("feedback", ""),
        "history": [f"human: {decision}"],
    }


def route_after_gate(state: ScoutState) -> str:
    """Send the run to commit, back to drafting, or to the bin."""
    decision = state.get("decision", "reject")
    if decision == "approve":
        return "commit"
    if decision == "edit" and state.get("revision", 1) < config.MAX_REVISIONS:
        return "draft"
    return "discard"


def _entry_markdown(state: ScoutState) -> str:
    """Render one approved item as the markdown that lands in the reading list."""
    item = state["item"]
    draft = state["draft"]
    bullets = "\n".join(f"- {point}" for point in draft.get("key_points", []))
    tags = " ".join(f"`{tag}`" for tag in draft.get("tags", []))
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (
        f"\n## {draft.get('headline', item['title'])}\n\n"
        f"{draft.get('why_it_matters', '')}\n\n"
        f"{bullets}\n\n"
        f"[{item['title']}]({item['url']})\n\n"
        f"> {item['source']} · score {state.get('score')} · {tags} · approved {stamp}\n"
    )


def commit_node(state: ScoutState) -> dict:
    """Append the approved entry to the reading list. The one real side effect.

    Runs only after a human said yes, and only once — the gate routes here exactly
    once per run, and a settled thread cannot be resumed again.
    """
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not config.READING_LIST.exists():
        config.READING_LIST.write_text(
            "# Reading list\n\nApproved overnight by night-scout.\n",
            encoding="utf-8",
        )
    with config.READING_LIST.open("a", encoding="utf-8") as handle:
        handle.write(_entry_markdown(state))

    return {
        "status": "committed",
        "result": f"added to {config.READING_LIST.name}",
        "history": ["commit: written to reading list"],
    }


def discard_node(state: ScoutState) -> dict:
    """Terminal for a rejected draft, or one that used up its revisions."""
    reason = "rejected" if state.get("decision") == "reject" else "out of revisions"
    return {
        "status": "discarded",
        "result": reason,
        "history": [f"discard: {reason}"],
    }
