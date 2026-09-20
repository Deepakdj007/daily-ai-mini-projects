"""One task, start to finish: the round loop that lets the toolbox grow.

Inputs:  a task, its family, an arm's policy, and a registry
Outputs: one results row, with every number the report needs

The mechanic this whole project is about is a single line below, and it is the
one the diagram should point at: the tools array is rebuilt inside the loop
rather than above it. Every other agent in this series builds its tool list
once, because its tool list cannot change. Here a tool written in round 1 is in
the array in round 2, and that difference is the entire feature.

Nothing here trusts what the model says it did. Reuse is counted from the tool
calls that actually happened, not from a field the model fills in, because a
model reporting on its own behaviour is the one witness with an interest.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from src import (cache, config, guard, llm, matcher, prompts, registry,
                 sandbox, smith, spec)


@dataclass
class Metrics:
    """What one task cost and what happened in it."""

    rounds: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    repairs: int = 0
    writes: int = 0
    registered: int = 0
    rejections: Counter = field(default_factory=Counter)
    notes: list[str] = field(default_factory=list)
    """Why each rejection happened, not just how many. The write ledger in the
    report is built from these, and a guard layer that fires often on honest
    tools shows up here before it shows up as a lost arm."""
    offered: list[list[str]] = field(default_factory=list)
    called: list[str] = field(default_factory=list)
    reused_from: str = ""
    answering_tool: str = ""
    stopped: str = ""
    retried: int = 0
    """How many calls came from a rung above the first of the retry ladder, and
    so ran at a non-zero temperature. A gate voids a run that drifted off
    temperature 0 too often: an eval that quietly re-rolls is not repeatable."""


def _tool_names(schemas: list[dict]) -> list[str]:
    return [s["function"]["name"] for s in schemas]


def _offer(reg: registry.Registry, policy, metrics: Metrics) -> list[dict]:
    """The tools array for this round.

    Deliberately not hoisted out of the loop. A tool registered last round is
    in this array; that is the self-extension mechanic, and it is one line.
    """
    offered = reg.schemas() if policy.tools else []
    if policy.write:
        offered = [*offered, spec.WRITE_TOOL]
    elif policy.inline:
        offered = [*offered, spec.RUN_PYTHON]
    return [*offered, spec.FINAL_ANSWER]


def _unwrap(value: object) -> str:
    """The answer, with one layer of JSON envelope taken off if it is there.

    Models sometimes put the whole call object into the field instead of the
    value, so the answer arrives as '{"value": "M"}' rather than 'M'. That is a
    correct answer in a wrapper, and grading it wrong would mark the model down
    for the shape of its envelope rather than for its reasoning.
    """
    text = str(value).strip()
    if not (text.startswith(("{", '"')) and text.endswith(("}", '"'))):
        return text
    try:
        inner = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(inner, str):
        # '"L"' is the answer L, JSON-encoded once more than it needed to be.
        return inner.strip()
    if isinstance(inner, dict):
        for key in (spec.ANSWER_FIELD, "answer", "result"):
            if key in inner:
                return str(inner[key]).strip()
        if len(inner) == 1:
            return str(next(iter(inner.values()))).strip()
    return text


def _run_inline(source: str) -> str:
    """The control's tool: guard it, run it once, keep nothing."""
    verdict = guard.guard(source)
    if not verdict.ok:
        return f"REJECTED ({verdict.layer}): {verdict.reason}"
    results, raw = sandbox.execute(source, [{}])
    row = results[0]
    if not row.get("ok"):
        return f"FAILED: {row.get('error', raw.outcome)}. Fix it and run it again."
    return json.dumps(row["value"], ensure_ascii=False, default=str)[:config.MAX_RESULT_CHARS]


def _dispatch(name: str, args: dict, *, reg: registry.Registry, policy, task: dict,
              family: dict, metrics: Metrics, round_no: int, arm: str) -> str:
    """Run one tool call and return the text the model reads."""
    if name == "write_tool":
        # The cap is enforced here rather than by dropping the schema. Taking
        # a tool out of the array mid-conversation reads to the model as a
        # glitch: it has three of its own write_tool calls in the history, so
        # it writes a fourth, and Groq rejects the whole request with a 400
        # naming a tool "not in request.tools" - which ends the task on a
        # transport error instead of on an answer. Refusing in code is just as
        # much a fact and costs one short tool message.
        if metrics.repairs >= config.MAX_REPAIR_ATTEMPTS:
            metrics.rejections["refused"] += 1
            return smith.refusal()
        metrics.writes += 1
        outcome = smith.attempt(args, task=task, family=family, policy=policy,
                                reg=reg, round_no=round_no, arm=arm)
        if outcome.ok:
            metrics.registered += 1
        else:
            metrics.rejections[outcome.stage] += 1
            label = f"{outcome.stage}/{outcome.guard_layer}" if outcome.guard_layer \
                else outcome.stage
            metrics.notes.append(f"{label}: {outcome.message.splitlines()[0][:150]}")
            if outcome.burns_repair:
                metrics.repairs += 1
        return outcome.message
    if name == "run_python":
        return _run_inline(spec.join_source(args.get("source")))

    metrics.tool_calls += 1
    metrics.called.append(name)
    tool = reg.get(name)
    if tool is not None:
        metrics.answering_tool = name
        # Reuse means a tool written on an EARLIER task, established from what
        # was called rather than from what was claimed.
        if tool.task_id and tool.task_id != task["id"]:
            metrics.reused_from = tool.task_id
    return reg.dispatch(name, args)


async def run_task(task: dict, family: dict, *, arm: str, policy,
                   reg: registry.Registry) -> dict:
    """Run one task under one arm and return its results row."""
    reg.begin_task(task["id"])
    metrics = Metrics()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": prompts.system_prompt(policy)},
        {"role": "user", "content": task["prompt"]},
    ]
    answer, outcome = "", "no_answer"

    for round_no in range(config.MAX_ROUNDS):
        offered = _offer(reg, policy, metrics)
        metrics.offered.append(_tool_names(offered))
        metrics.rounds = round_no + 1
        budget = config.MAX_COMPLETION_TOKENS[
            "write" if any(s["function"]["name"] in ("write_tool", "run_python")
                           for s in offered) else "answer"]
        key = cache.key(config.CHAT_MODEL, {
            "v": config.PROMPT_VERSION, "arm": arm, "task": task["id"],
            "round": round_no, "messages": messages, "tools": _tool_names(offered)})

        completion = await llm.complete(messages, tools=offered, cache_key=key,
                                        max_completion_tokens=budget)
        metrics.llm_calls += 1
        metrics.prompt_tokens += completion.usage.prompt_tokens
        metrics.completion_tokens += completion.usage.completion_tokens
        metrics.retried += 1 if completion.attempt else 0
        if completion.finish_reason == "tpd_exhausted":
            metrics.stopped = "tpd_exhausted"
            outcome = "error"
            break
        if not completion.ok:
            outcome = "error"
            answer = (completion.error or completion.finish_reason)[:300]
            break

        messages.append(llm.assistant_message(completion))
        if not completion.tool_calls:
            # The model answered in prose instead of calling final_answer. Take
            # the text rather than throwing the task away; the report counts
            # how often this happens.
            answer, outcome = completion.text, "text_answer"
            break

        finished = False
        for call in completion.tool_calls:
            try:
                args = json.loads(call["arguments"] or "{}")
            except json.JSONDecodeError as exc:
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": f"ERROR: arguments were not valid JSON ({exc})"})
                continue
            if call["name"] == spec.ANSWER_TOOL:
                answer = _unwrap(args.get(spec.ANSWER_FIELD, ""))
                outcome, finished = "answered", True
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": "noted"})
                break
            text = _dispatch(call["name"], args, reg=reg, policy=policy, task=task,
                             family=family, metrics=metrics, round_no=round_no, arm=arm)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": text})
        if finished:
            break

    if outcome == "no_answer" and metrics.repairs >= config.MAX_REPAIR_ATTEMPTS:
        outcome = "gave_up"
    correct = bool(answer) and matcher.matches(answer, task["answer"], family["compare"])
    return {
        "task_id": task["id"], "family": task["family"], "member": task["member"],
        "arm": arm, "answer": answer, "gold": task["answer"],
        "correct": correct,
        "outcome": "correct" if correct else ("wrong" if outcome in
                                              ("answered", "text_answer") else outcome),
        "rounds": metrics.rounds, "llm_calls": metrics.llm_calls,
        "tool_calls": metrics.tool_calls,
        "prompt_tokens": metrics.prompt_tokens,
        "completion_tokens": metrics.completion_tokens,
        "writes": metrics.writes, "registered": metrics.registered,
        "repairs": metrics.repairs, "rejections": dict(metrics.rejections),
        "notes": metrics.notes,
        "offered": metrics.offered, "called": metrics.called,
        "reused_from": metrics.reused_from,
        "reused": bool(metrics.reused_from),
        "answering_tool": metrics.answering_tool,
        "registry": reg.stamp(), "stopped": metrics.stopped,
        "retried": metrics.retried,
    }
