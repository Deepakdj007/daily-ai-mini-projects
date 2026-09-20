"""One task, start to finish: the round loop that lets the toolbox grow.

Inputs:  a task, its family, and a registry
Outputs: what happened - the answer, the tools offered each round, the rejections

The mechanic this whole project is about is one line below, and it is the line
a diagram should point at: the tools array is rebuilt INSIDE the loop rather
than above it. Every other tool-calling agent builds its tool list once,
because its tool list cannot change. Here a tool written in round 1 is in the
array in round 2, and that difference is the entire feature.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from src import config, llm, prompts, registry, smith


@dataclass
class Run:
    """What one task cost and what happened in it."""

    answer: str = ""
    correct: bool = False
    outcome: str = "no answer"
    rounds: int = 0
    completion_tokens: int = 0
    writes: int = 0
    registered: int = 0
    repairs: int = 0
    reused: str = ""
    offered: list[list[str]] = field(default_factory=list)
    called: list[str] = field(default_factory=list)
    rejections: list[str] = field(default_factory=list)


def matches(got: object, want: object) -> bool:
    """Case- and space-insensitive comparison. Both answers are compact strings."""
    clean = lambda value: " ".join(str(value).split()).strip().upper()   # noqa: E731
    return bool(clean(got)) and clean(got) == clean(want)


def unwrap(value: object) -> str:
    """The answer, with one layer of JSON envelope taken off if it is there.

    Models sometimes put the whole call object into the field, so the answer
    arrives as '{"value": "M"}' or '"M"' rather than 'M'. That is a correct
    answer in a wrapper, and grading it wrong would mark the model down for the
    shape of its envelope.
    """
    text = str(value).strip()
    if not (text.startswith(("{", '"')) and text.endswith(("}", '"'))):
        return text
    try:
        inner = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(inner, str):
        return inner.strip()
    if isinstance(inner, dict):
        for key in (smith.ANSWER_FIELD, "answer", "result"):
            if key in inner:
                return str(inner[key]).strip()
    return text


def _dispatch(name: str, args: dict, *, reg: registry.Registry, task: dict,
              family: dict, run: Run) -> str:
    """Run one tool call and return the text the model reads."""
    if name == "write_tool":
        # The repair cap is enforced HERE rather than by dropping the schema
        # from the array. Taking a tool away mid-conversation reads to the
        # model as a glitch: it has three of its own write_tool calls in the
        # history, so it writes a fourth, and Groq rejects the whole request
        # with a 400 naming a tool "not in request.tools" - which ends the task
        # on a transport error instead of on an answer.
        if run.repairs >= config.MAX_REPAIR_ATTEMPTS:
            return smith.refusal()
        run.writes += 1
        outcome = smith.attempt(args, task=task, family=family, reg=reg)
        if outcome.ok:
            run.registered += 1
        else:
            run.rejections.append(f"{outcome.stage}: {outcome.message.splitlines()[0][:130]}")
            if outcome.burns_repair:
                run.repairs += 1
        return outcome.message

    run.called.append(name)
    tool = reg.get(name)
    if tool is not None and tool.task_id and tool.task_id != task["id"]:
        # Reuse, established from what was actually called rather than from
        # anything the model claims about itself.
        run.reused = tool.task_id
    return reg.dispatch(name, args)


async def solve(task: dict, family: dict, reg: registry.Registry) -> Run:
    """Run one task and return what happened."""
    run = Run()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": prompts.system_prompt()},
        {"role": "user", "content": task["prompt"]},
    ]

    for round_no in range(config.MAX_ROUNDS):
        # Rebuilt every round, deliberately NOT hoisted out of the loop. A tool
        # registered last round is in this array. That is the whole mechanic.
        offered = [*reg.schemas(), smith.WRITE_TOOL, smith.SUBMIT_ANSWER]
        run.offered.append([s["function"]["name"] for s in offered])
        run.rounds = round_no + 1

        completion = await llm.complete(messages, offered)
        run.completion_tokens += completion.completion_tokens
        if not completion.ok:
            run.outcome, run.answer = "error", (completion.error or "")[:200]
            break

        messages.append(llm.assistant_message(completion))
        if not completion.tool_calls:
            # The model answered in text instead of calling submit_answer, and
            # it often writes the call's JSON body out verbatim when it does,
            # so this goes through the same unwrapping.
            run.answer, run.outcome = unwrap(completion.text), "answered in prose"
            break

        finished = False
        for call in completion.tool_calls:
            # gpt-oss sometimes emits "functions.submit_answer" rather than
            # "submit_answer" - the namespace from the harmony format it was
            # trained on, leaking through. No tool name here may contain a dot
            # (the registry forbids it), so taking the last segment is safe,
            # and without this the answer call falls through to the registry
            # and wastes a round on "there is no tool called that".
            name = call["name"].rsplit(".", 1)[-1]
            try:
                args = json.loads(call["arguments"] or "{}")
            except json.JSONDecodeError as exc:
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": f"ERROR: arguments were not valid JSON ({exc})"})
                continue
            if name == smith.ANSWER_TOOL:
                run.answer = unwrap(args.get(smith.ANSWER_FIELD, ""))
                run.outcome, finished = "answered", True
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": "noted"})
                break
            text = _dispatch(name, args, reg=reg, task=task, family=family, run=run)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": text})
        if finished:
            break

    run.correct = matches(run.answer, task["answer"])
    if run.correct:
        run.outcome = "correct"
    elif run.outcome == "answered":
        run.outcome = "wrong"
    return run


if __name__ == "__main__":
    assert matches("kxmsgrsi", "KXMSGRSI") and matches(" V ", "V")
    assert not matches("", "V") and not matches("K", "V")
    assert unwrap('{"value": "M"}') == "M"
    assert unwrap('"L"') == "L"
    assert unwrap("KXMSGRSI") == "KXMSGRSI"
    assert unwrap('{"answer": 4}') == "4"
    # The harmony-format namespace has to come off, or the answer call is
    # mistaken for a tool nobody registered.
    assert "functions.submit_answer".rsplit(".", 1)[-1] == smith.ANSWER_TOOL
    assert "gstin_check".rsplit(".", 1)[-1] == "gstin_check"
    print("OK - agent (grading and unwrapping; the loop needs an API key)")
