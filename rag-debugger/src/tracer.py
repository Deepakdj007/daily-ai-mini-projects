"""A tiny span recorder: every pipeline stage becomes a timed, saved Step.

Inputs:  stage names and whatever each stage wants to record
Outputs: a Trace object, and a JSON file per run in output/traces/

This is the same idea as an OpenTelemetry span or a Langfuse observation,
cut down to what fits on one screen: a name, a duration and a summary.
"""

import json
import re
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from src import config


@dataclass
class Step:
    """One stage of the pipeline: what it was called, how long it took, what it saw."""

    name: str
    ms: float = 0.0
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trace:
    """Everything one query did, in order. Rows and diagnosis are filled in by the pipeline."""

    query: str
    settings: dict[str, Any]
    steps: list[Step] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    diagnosis: dict[str, Any] = field(default_factory=dict)
    answer: str = ""
    saved_to: str = ""

    @contextmanager
    def step(self, name: str) -> Iterator[Step]:
        """Time a block of code and record it as a Step, even if the block raises."""
        record = Step(name=name)
        started = time.perf_counter()
        try:
            yield record
        finally:
            record.ms = round((time.perf_counter() - started) * 1000, 1)
            self.steps.append(record)

    def save(self, directory: Path = config.TRACE_DIR) -> Path:
        """Write the trace as JSON so it can be reopened, diffed or shipped to a UI later."""
        directory.mkdir(parents=True, exist_ok=True)
        # A question in Hindi has no a-z characters at all, so fall back to a fixed name.
        slug = re.sub(r"[^a-z0-9]+", "-", self.query.lower()).strip("-")[:40] or "query"
        path = directory / f"{datetime.now():%Y%m%d-%H%M%S}-{slug}.json"
        self.saved_to = str(path.relative_to(config.ROOT_DIR))
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")
        return path


if __name__ == "__main__":
    trace = Trace(query="self check", settings={"top_k": 3})
    with trace.step("sleep") as s:
        time.sleep(0.05)
        s.summary["slept"] = "50 ms"
    assert trace.steps[0].ms >= 50, trace.steps
    print(f"step recorded: {trace.steps[0]}")
    print(f"saved to {trace.save(config.ROOT_DIR / 'output' / 'selfcheck')}")
