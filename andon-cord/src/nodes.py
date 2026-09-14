"""The three nodes that do the work: lead, researcher, writer.

None of these calls interrupt(). Pausing lives in src/cord.py, because a node
that pauses re-runs from its first line when the run resumes — and re-running a
web search and an LLM call to get back to the same pause is pure waste.

Inputs:  ResearchState (lead, writer) or ResearchTask (researcher)
Outputs: partial state updates
"""

from src import config, prompts
from src.search import web_search
from src.state import Finding, Plan, ResearchState, ResearchTask


def _format_clarifications(clarifications: list[dict]) -> str:
    """Render answered questions for a prompt, or nothing when there are none."""
    if not clarifications:
        return ""
    lines = "\n".join(
        f"- You asked: {c['text']}\n  The human answered: {c['answer']}"
        for c in clarifications
    )
    return f"\nAnswers you already have from the human:\n{lines}\n"


def lead_node(state: ResearchState) -> dict:
    """Read the brief, split it into research questions, maybe ask one question.

    Args:
        state: needs `brief`, and `clarifications` once the human has answered.

    Returns:
        interpretation, sub_questions, needs_human and question.
    """
    prompt = prompts.LEAD_PROMPT.format(
        brief=state["brief"],
        clarifications=_format_clarifications(state.get("clarifications", [])),
        n=config.NUM_RESEARCHERS,
        ask_rule=prompts.ASK_RULE,
    )
    plan: Plan = config.structured(Plan, config.MAX_TOKENS_LEAD).invoke(prompt)
    return {
        "interpretation": plan.interpretation,
        "sub_questions": plan.sub_questions[: config.NUM_RESEARCHERS],
        "needs_human": plan.needs_human,
        "question": plan.question.strip(),
    }


def _format_results(results: list[dict]) -> str:
    """Number the search results so the model can cite them as [1], [2], ..."""
    if not results:
        return "(no results)"
    return "\n\n".join(
        f"[{i}] {r['title']}\n{r['url']}\n{r['snippet']}"
        for i, r in enumerate(results, 1)
    )


def researcher_node(task: ResearchTask) -> dict:
    """Search the web for one sub-question and write a cited answer.

    Args:
        task: the Send payload. This is the whole input — a node dispatched by
            Send does not see the shared state.

    Returns:
        A one-element `findings` list, appended by the reducer.

    This must never raise. Both researchers run in the same superstep, and one
    exception fails the superstep for both.
    """
    results = web_search(task["question"])
    prompt = prompts.RESEARCHER_PROMPT.format(
        brief=task["brief"],
        interpretation=task["interpretation"],
        clarifications=_format_clarifications(task["clarifications"]),
        question=task["question"],
        results=_format_results(results),
        ask_rule=prompts.ASK_RULE,
    )
    model = config.structured(Finding, config.MAX_TOKENS_RESEARCHER)
    failed = ""
    try:
        finding: Finding = model.invoke(prompt)
        summary, needs_human, ask = finding.summary, finding.needs_human, finding.question
    except Exception as exc:  # noqa: BLE001 - a dead researcher must not kill the run
        summary, needs_human, ask = "The sources do not settle this.", False, ""
        failed = f"{type(exc).__name__}: {exc}"[:200]
    return {
        "findings": [
            {
                "index": task["index"],
                "question": task["question"],
                "summary": summary,
                "needs_human": needs_human,
                "ask": ask.strip(),
                "sources": results,
                "failed": failed,
                "n_results": len(results),
            }
        ]
    }


def _format_findings(findings: list[dict]) -> str:
    """Render every finding with its own numbered source list for the writer."""
    blocks: list[str] = []
    for finding in sorted(findings, key=lambda f: f["index"]):
        listed = "\n".join(
            f"  [{i}] {s['title']} -- {s['url']}"
            for i, s in enumerate(finding["sources"], 1)
        )
        blocks.append(
            f"Q: {finding['question']}\nA: {finding['summary']}\n"
            f"Sources for this answer:\n{listed or '  (none)'}"
        )
    return "\n\n".join(blocks)


def _scope_line(state: ResearchState) -> str:
    """State what the team settled on, so a reader can check it was the right job."""
    parts = [state.get("interpretation", "")]
    for clarification in state.get("clarifications", []):
        parts.append(f"{clarification['text']} -> {clarification['answer']}")
    return " | ".join(p for p in parts if p)


def _sources_section(findings: list[dict], report: str) -> str:
    """Append a bibliography of the URLs the report actually links to.

    The team fetches far more results than the writer uses, and DuckDuckGo
    returns the occasional unrelated hit. Listing every URL fetched pads the
    bibliography with sources no claim rests on.
    """
    seen: set[str] = set()
    cited: list[dict] = []
    for finding in findings:
        for source in finding["sources"]:
            url = source["url"]
            if url and url not in seen and url in report:
                seen.add(url)
                cited.append(source)
    if not cited:
        return ""
    lines = "\n".join(f"{i}. [{s['title']}]({s['url']})" for i, s in enumerate(cited, 1))
    return f"\n\n---\n\n## Sources\n\n{lines}\n"


def writer_node(state: ResearchState) -> dict:
    """Turn the findings into one cited report headed by the agreed scope."""
    findings = state.get("findings", [])
    scope = _scope_line(state)
    prompt = prompts.WRITER_PROMPT.format(
        brief=state["brief"], scope=scope or "(none agreed)",
        findings=_format_findings(findings),
    )
    body = config.make_llm(config.MAX_TOKENS_WRITER).invoke(prompt).content
    report = f"{body}\n\n> **Scope:** {scope}\n" if scope else body
    return {"report": report + _sources_section(findings, report)}
