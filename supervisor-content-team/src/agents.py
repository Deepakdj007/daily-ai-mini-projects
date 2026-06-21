"""The four specialist nodes: researcher, writer, editor, SEO.

Each is a LangGraph node: it takes the shared ContentState, does one job with
an LLM (the researcher also calls the Tavily search tool), and returns a partial
state dict. Every node also appends a one-line entry to `history` so the run is
easy to follow.

Inputs:  ContentState.
Outputs: partial ContentState updates merged by the graph.
"""

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_tavily import TavilySearch

from src.config import MODEL_WORKER, TAVILY_MAX_RESULTS, make_llm
from src.prompts import EDITOR_PROMPT, RESEARCHER_PROMPT, SEO_PROMPT, WRITER_PROMPT
from src.state import ContentState

# One worker model shared by all specialists; the search tool reads its key
# from the TAVILY_API_KEY environment variable automatically.
_llm = make_llm(MODEL_WORKER, temperature=0.4)
_search = TavilySearch(max_results=TAVILY_MAX_RESULTS, topic="general")


def _ask(system: str, user: str, temperature: float = 0.4) -> str:
    """Send one system+user turn to the worker model and return the text.

    Gemini 3.x can return `.content` as a list of content blocks rather than a
    plain string, so use the message's `.text` accessor to get clean text.
    """
    llm = make_llm(MODEL_WORKER, temperature=temperature)
    reply = llm.invoke([SystemMessage(system), HumanMessage(user)])
    return reply.text


def researcher_node(state: ContentState) -> dict:
    """Search the web for the topic and synthesize a research brief."""
    topic = state["topic"]
    raw = _search.invoke({"query": topic})
    # TavilySearch returns a dict with a "results" list of {title, url, content}.
    snippets = "\n\n".join(
        f"- {r.get('title', '')}: {r.get('content', '')}"
        for r in raw.get("results", [])
    )
    brief = _ask(RESEARCHER_PROMPT, f"Topic: {topic}\n\nSearch results:\n{snippets}")
    return {"research": brief, "history": ["researcher: gathered and synthesized sources"]}


def writer_node(state: ContentState) -> dict:
    """Draft (or rewrite) the article from the research brief."""
    topic = state["topic"]
    research = state.get("research", "")
    revisions = state.get("revision_count", 0)

    if state.get("draft") and state.get("editor_verdict") == "revision":
        # A rewrite pass: fold the editor's feedback into the instructions.
        feedback = state.get("editor_feedback", "")
        edited = state.get("edited", "")
        user = (
            f"Topic: {topic}\n\nResearch brief:\n{research}\n\n"
            f"Editor's current version:\n{edited}\n\n"
            f"The editor asked for a change: {feedback}\n\n"
            "Rewrite the full article addressing that feedback."
        )
        revisions += 1
        note = f"writer: rewrote draft (revision {revisions})"
    else:
        user = f"Topic: {topic}\n\nResearch brief:\n{research}\n\nWrite the article."
        note = "writer: produced first draft"

    draft = _ask(WRITER_PROMPT, user, temperature=0.6)
    return {"draft": draft, "revision_count": revisions, "history": [note]}


def editor_node(state: ContentState) -> dict:
    """Tighten the draft and emit an APPROVED / REVISION verdict."""
    draft = state.get("draft", "")
    result = _ask(EDITOR_PROMPT, f"Draft to edit:\n\n{draft}", temperature=0.3)

    # The verdict is on the final line. Read it, then strip it so the verdict
    # marker never leaks into the published article.
    lines = result.strip().splitlines()
    last_line = lines[-1].strip().upper()
    if last_line.startswith("REVISION") and state.get("revision_count", 0) == 0:
        verdict = "revision"
        feedback = lines[-1].strip()  # "REVISION: <reason>"
        note = "editor: requested one revision"
    elif last_line.startswith(("APPROVED", "REVISION")):
        verdict, feedback = "approved", ""
        note = "editor: approved the article"
    else:
        # No recognizable verdict line; keep the whole result and approve.
        return {"edited": result, "editor_verdict": "approved", "editor_feedback": "",
                "history": ["editor: approved the article"]}

    article = "\n".join(lines[:-1]).strip()
    return {"edited": article, "editor_verdict": verdict, "editor_feedback": feedback,
            "history": [note]}


def seo_node(state: ContentState) -> dict:
    """Produce the SEO package as a parsed dict."""
    article = state.get("edited") or state.get("draft", "")
    raw = _ask(SEO_PROMPT, f"Article:\n\n{article}", temperature=0.2)
    seo = _parse_seo(raw)
    return {"seo": seo, "history": ["seo: generated title, meta, keywords, slug"]}


def _parse_seo(raw: str) -> dict:
    """Parse the SEO model's JSON, tolerating ```json fences."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{") : text.rfind("}") + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"title": "", "meta_description": "", "keywords": [], "slug": "article"}
