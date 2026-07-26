"""The research team itself: five steps wired together by events.

The shape of the graph:

    StartEvent -> plan -> (fan out N) -> research -> (fan in) -> gather
                                            ^                      |
                                            |                      v
                                            +----- reflect <-------+
                                                      |
                                                      v
                                                    write -> StopEvent

Fan-out is `ctx.send_event` in a loop. Fan-in is `ctx.collect_events`, which
returns None every time it is called until the last expected event arrives.
The reflector can send the team back out once, which is what makes this a loop
rather than a straight line.

Inputs:  a topic string, passed as `flow.run(topic=...)`.
Outputs: StopEvent carrying the finished markdown report.
"""

from workflows import Context, Workflow, step
from workflows.events import StartEvent, StopEvent
from workflows.retry_policy import (
    retry_policy,
    stop_after_attempt,
    wait_exponential,
)

from src import config, prompts
from src.events import (
    Critique,
    Finding,
    FindingEvent,
    Plan,
    ProgressEvent,
    ReflectEvent,
    Source,
    SubQuestionEvent,
    WriteEvent,
)
from src.llm import astructured, atext
from src.search import web_search

# Search and model calls both fail transiently on a free tier. Retry the whole
# research step rather than letting one flaky sub-question kill the run.
RESEARCH_RETRY = retry_policy(
    wait=wait_exponential(multiplier=1, exp_base=2, max=20),
    stop=stop_after_attempt(3),
)


def _format_results(sources: list[Source]) -> str:
    """Number the search results so the model can cite them as [1], [2], ..."""
    if not sources:
        return "(no results)"
    return "\n\n".join(
        f"[{i}] {s.title}\n{s.url}\n{s.snippet}" for i, s in enumerate(sources, 1)
    )


def _format_findings(findings: list[Finding], with_sources: bool = False) -> str:
    """Render collected findings for the reflector or the writer.

    Each researcher cites its own results as [1], [2], ... so those numbers only
    mean anything next to that researcher's own source list. The writer gets
    with_sources=True so it can resolve each bracket back to a real URL; the
    reflector only judges coverage and does not need them.
    """
    blocks: list[str] = []
    for finding in findings:
        block = f"Q: {finding.question}\nA: {finding.summary}"
        if with_sources:
            listed = "\n".join(
                f"  [{i}] {s.title} -- {s.url}"
                for i, s in enumerate(finding.sources, 1)
            )
            block += f"\nSources for this answer:\n{listed or '  (none)'}"
        blocks.append(block)
    return "\n\n".join(blocks)


class DeepResearchWorkflow(Workflow):
    """A planner, N parallel researchers, a reflector, and a writer."""

    def __init__(self, **kwargs) -> None:
        """Build the workflow and its two LLM clients."""
        super().__init__(**kwargs)
        self.fast_llm = config.make_llm(config.MODEL_FAST, temperature=0.2)
        self.writer_llm = config.make_llm(config.MODEL_WRITER, temperature=0.4)

    @step
    async def plan(self, ctx: Context, ev: StartEvent) -> SubQuestionEvent | None:
        """Split the topic into sub-questions and fan them out to the researchers.

        Note the return type. This step hands work off with ctx.send_event and
        then returns None, but the annotation must still declare SubQuestionEvent
        or graph validation rejects the workflow.
        """
        topic: str = ev.topic
        await ctx.store.set("topic", topic)
        await ctx.store.set("round", 1)
        await ctx.store.set("findings", [])
        await ctx.store.set("asked", [])

        ctx.write_event_to_stream(
            ProgressEvent(agent="planner", msg=f"Planning research on: {topic}")
        )

        plan = await astructured(
            self.fast_llm,
            prompts.PLANNER_PROMPT.format(topic=topic, n=config.NUM_SUB_QUESTIONS),
            Plan,
        )
        questions = plan.sub_questions[: config.NUM_SUB_QUESTIONS]

        # gather needs to know how many findings to wait for before it fires.
        await ctx.store.set("pending", len(questions))
        await ctx.store.set("asked", list(questions))

        for i, question in enumerate(questions, 1):
            ctx.write_event_to_stream(
                ProgressEvent(agent="planner", msg=f"Q{i}: {question}", style="dim")
            )
            ctx.send_event(SubQuestionEvent(question=question, index=i, round=1))
        return None

    @step(num_workers=config.RESEARCH_WORKERS, retry_policy=RESEARCH_RETRY)
    async def research(self, ctx: Context, ev: SubQuestionEvent) -> FindingEvent:
        """Search the web for one sub-question and summarise what came back.

        num_workers is what makes this concurrent: the engine keeps that many
        copies of this step in flight, so the searches and model calls overlap
        instead of queueing. Set RESEARCH_WORKERS=1 to feel the difference.
        """
        ctx.write_event_to_stream(
            ProgressEvent(
                agent=f"research-{ev.index}",
                msg=f"searching: {ev.question}",
                style="yellow",
            )
        )
        sources = await web_search(ev.question, config.SEARCH_RESULTS)

        summary = await atext(
            self.fast_llm,
            prompts.RESEARCHER_PROMPT.format(
                question=ev.question, results=_format_results(sources)
            ),
        )
        ctx.write_event_to_stream(
            ProgressEvent(
                agent=f"research-{ev.index}",
                msg=f"done ({len(sources)} sources)",
                style="green",
            )
        )
        finding = Finding(
            question=ev.question, summary=summary, sources=sources, round=ev.round
        )
        return FindingEvent(finding=finding)

    @step
    async def gather(self, ctx: Context, ev: FindingEvent) -> ReflectEvent | None:
        """Wait for every researcher in this round, then hand over to the reflector.

        collect_events buffers events and returns None until it has one of every
        type in `expected`. Asking for the same type N times is how you wait for
        N parallel results.
        """
        pending: int = await ctx.store.get("pending", default=0)
        collected = ctx.collect_events(ev, [FindingEvent] * pending)
        if collected is None:
            return None

        findings: list[dict] = await ctx.store.get("findings", default=[])
        findings.extend(e.finding.model_dump() for e in collected)
        await ctx.store.set("findings", findings)

        current_round: int = await ctx.store.get("round", default=1)
        ctx.write_event_to_stream(
            ProgressEvent(
                agent="gather",
                msg=f"round {current_round}: collected {len(collected)} findings "
                f"({len(findings)} total)",
                style="magenta",
            )
        )
        return ReflectEvent(round=current_round)

    @step
    async def reflect(
        self, ctx: Context, ev: ReflectEvent
    ) -> SubQuestionEvent | WriteEvent | None:
        """Judge coverage. Either send the team back out once, or start writing."""
        topic: str = await ctx.store.get("topic")
        raw: list[dict] = await ctx.store.get("findings", default=[])
        findings = [Finding(**f) for f in raw]

        # Hard stop. Without this the reflector could keep asking for one more round.
        if ev.round >= config.MAX_ROUNDS:
            ctx.write_event_to_stream(
                ProgressEvent(
                    agent="reflector",
                    msg=f"round cap ({config.MAX_ROUNDS}) reached - writing now",
                    style="magenta",
                )
            )
            return WriteEvent()

        critique = await astructured(
            self.fast_llm,
            prompts.REFLECTOR_PROMPT.format(
                topic=topic,
                findings=_format_findings(findings),
                max_gaps=config.MAX_GAP_QUESTIONS,
            ),
            Critique,
        )
        gaps = critique.gaps[: config.MAX_GAP_QUESTIONS]

        if critique.covered or not gaps:
            ctx.write_event_to_stream(
                ProgressEvent(
                    agent="reflector",
                    msg=f"coverage OK - {critique.reasoning}",
                    style="magenta",
                )
            )
            return WriteEvent()

        ctx.write_event_to_stream(
            ProgressEvent(
                agent="reflector",
                msg=f"gaps found ({len(gaps)}) - {critique.reasoning}",
                style="red",
            )
        )

        # Same fan-out as the planner, one round later. Update the pending count
        # before sending, or gather will wait for the wrong number of findings.
        next_round = ev.round + 1
        await ctx.store.set("round", next_round)
        await ctx.store.set("pending", len(gaps))
        asked: list[str] = await ctx.store.get("asked", default=[])
        await ctx.store.set("asked", asked + gaps)

        offset = len(asked)
        for i, question in enumerate(gaps, offset + 1):
            ctx.write_event_to_stream(
                ProgressEvent(agent="reflector", msg=f"Q{i}: {question}", style="dim")
            )
            ctx.send_event(
                SubQuestionEvent(question=question, index=i, round=next_round)
            )
        return None

    @step
    async def write(self, ctx: Context, ev: WriteEvent) -> StopEvent:
        """Turn every finding into one cited markdown report."""
        topic: str = await ctx.store.get("topic")
        raw: list[dict] = await ctx.store.get("findings", default=[])
        findings = [Finding(**f) for f in raw]

        ctx.write_event_to_stream(
            ProgressEvent(
                agent="writer",
                msg=f"writing report from {len(findings)} findings",
                style="blue",
            )
        )

        sources = _dedupe_sources(findings)
        report = await atext(
            self.writer_llm,
            prompts.WRITER_PROMPT.format(
                topic=topic, findings=_format_findings(findings, with_sources=True)
            ),
        )
        return StopEvent(result=report + _sources_section(sources, report))


def _dedupe_sources(findings: list[Finding]) -> list[Source]:
    """Flatten every finding's sources into one list, keeping first-seen order."""
    seen: set[str] = set()
    unique: list[Source] = []
    for finding in findings:
        for source in finding.sources:
            if source.url and source.url not in seen:
                seen.add(source.url)
                unique.append(source)
    return unique


def _sources_section(sources: list[Source], report: str) -> str:
    """Append a bibliography of the sources the report actually links to.

    Researchers collect far more results than the writer ends up using, and
    DuckDuckGo returns the occasional unrelated hit. Listing every URL that was
    fetched pads the bibliography with sources no claim rests on, so keep only
    the ones whose URL appears in the finished report.
    """
    cited = [s for s in sources if s.url in report]
    if not cited:
        return ""
    lines = "\n".join(f"{i}. [{s.title}]({s.url})" for i, s in enumerate(cited, 1))
    return f"\n\n---\n\n## Sources\n\n{lines}\n"
