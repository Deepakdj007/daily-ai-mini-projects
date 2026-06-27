"""The three Reflexion nodes plus the loop's routing function.

generator_node writes (or rewrites) the email. critic_node grades it on the
cheap model and decides pass/fail. route_after_critic is the conditional that
either loops back for another revision or sends control to the adjudicator.
adjudicator_node runs once on the expensive reasoning model for a closing verdict.

Inputs:  ReflexionState (partial dicts merged by LangGraph).
Outputs: partial-state dicts; route_after_critic returns the next node name.
"""

from langchain_core.messages import HumanMessage, SystemMessage

from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

from src.config import (
    MAX_REVISIONS,
    MODEL_ADJUDICATOR,
    MODEL_ADJUDICATOR_FALLBACK,
    MODEL_CRITIC,
    MODEL_GENERATOR,
    PASS_THRESHOLD,
    make_llm,
)
from src.prompts import (
    ADJUDICATOR_SYSTEM,
    CRITIC_SYSTEM,
    GENERATOR_SYSTEM,
    REVISION_TEMPLATE,
)
from src.state import Critique, ReflexionState


def _text(response) -> str:
    """Pull plain text out of an LLM response.

    Gemini 3.x can return `content` as a list of typed blocks (text + reasoning)
    instead of a bare string, so handle both shapes.
    """
    content = response.content
    if isinstance(content, str):
        return content.strip()
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts).strip()


def generator_node(state: ReflexionState) -> dict:
    """Write the first draft, or rewrite the previous one around the critique.

    On revision 0 there is no feedback yet, so the generator works straight from
    the task. After that it gets the prior draft plus the single fix the critic
    asked for, and rewrites the whole email.
    """
    llm = make_llm(MODEL_GENERATOR, temperature=0.7)
    revision = state.get("revision", 0)

    if revision == 0:
        user = f"Write a cold outreach email for this brief:\n\n{state['task']}"
    else:
        user = REVISION_TEMPLATE.format(
            draft=state["draft"],
            score=state["score"],
            feedback=state["feedback"],
        )

    response = llm.invoke(
        [SystemMessage(content=GENERATOR_SYSTEM), HumanMessage(content=user)]
    )
    draft = _text(response)

    return {
        "draft": draft,
        "revision": revision + 1,
        "history": [f"generator: produced draft #{revision + 1}"],
    }


def critic_node(state: ReflexionState) -> dict:
    """Grade the current draft on the cheap model and record the score.

    Structured output forces a full rubric verdict every time. The score is
    appended to `scores` (additive reducer) so the plot gets one point per
    revision. `passed` is recomputed from the threshold so the critic can't
    pass a low-scoring draft.
    """
    llm = make_llm(MODEL_CRITIC, temperature=0.0)
    grader = llm.with_structured_output(Critique)

    system = CRITIC_SYSTEM.format(threshold=PASS_THRESHOLD)
    user = f"Grade this cold email:\n\n{state['draft']}"
    verdict: Critique = grader.invoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )

    passed = verdict.score >= PASS_THRESHOLD
    return {
        "score": verdict.score,
        "scores": [verdict.score],
        "feedback": verdict.feedback,
        "passed": passed,
        "history": [
            f"critic: scored {verdict.score}/10 "
            f"(hook {verdict.hook}, spec {verdict.specificity}, "
            f"clarity {verdict.clarity}, cta {verdict.cta}, "
            f"brevity {verdict.brevity}) -> {'PASS' if passed else 'revise'}"
        ],
    }


def route_after_critic(state: ReflexionState) -> str:
    """Loop back to the generator, or move on to the final adjudicator.

    Two exits keep the loop finite: a passing score, or hitting MAX_REVISIONS.
    Either way the run ends with exactly one adjudicator pass.
    """
    if state.get("passed") or state.get("revision", 0) >= MAX_REVISIONS:
        return "adjudicator"
    return "generator"


def adjudicator_node(state: ReflexionState) -> dict:
    """Run the pro reasoning model once for a closing human-style verdict.

    The pro tier is billing-only on the Gemini API, so if that call is blocked
    (404 not-found or 429 quota-exhausted) we fall back to the strongest free
    model and note which judge actually ran.
    """
    messages = [
        SystemMessage(content=ADJUDICATOR_SYSTEM),
        HumanMessage(
            content=(
                f"Brief:\n{state['task']}\n\n"
                f"Final email after {state.get('revision', 0)} revision(s), "
                f"critic score {state.get('score', 0)}/10:\n\n{state['draft']}"
            )
        ),
    ]

    model_used = MODEL_ADJUDICATOR
    try:
        response = make_llm(MODEL_ADJUDICATOR, temperature=0.2).invoke(messages)
    except ChatGoogleGenerativeAIError:
        # Pro tier unavailable on this key — degrade to the free model.
        model_used = MODEL_ADJUDICATOR_FALLBACK
        response = make_llm(MODEL_ADJUDICATOR_FALLBACK, temperature=0.2).invoke(messages)

    return {
        "final_verdict": _text(response),
        "verdict_model": model_used,
        "history": [f"adjudicator: delivered final verdict ({model_used})"],
    }
