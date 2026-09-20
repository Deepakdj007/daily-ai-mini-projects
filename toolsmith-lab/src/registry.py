"""The toolbox: what the agent can call, including what it wrote itself.

Inputs:  the generated/ directory, and tools handed over by src/smith.py
Outputs: Groq function schemas, and the result of calling one

A registered tool is never imported into this process. Every call goes back
through the sandbox, for four reasons in order of weight.

The guard is a static pass and the sandbox is the enforcement. AST analysis
cannot see through everything, so running model-written code in-process would
make every bypass a live bypass against a process holding the API key, the
spend ledger and the results file. In a subprocess a bypass is a crash.

A tool is validated by being run in the sandbox. If it were later called by
some other mechanism, the thing that passed the checks would not be the thing
answering, and a held-out pass would stop being a statement about it.

Loading the registry must never execute anything. With persist on, generated/
is read at startup, possibly days later. `import generated.gstin_check` would
run module-level code before any check looked at it; reading text cannot.

And the cost is nothing: an interpreter starts in about a tenth of a second,
against multi-second model latency, roughly twice per task.

The honest price of that decision: a subprocess keeps no state between calls,
so every generated tool has to be a pure function of its arguments. The system
prompt says so. It suits this corpus, and it is a real limit of the design.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src import config, sandbox, tools

# Groq accepts ^[a-zA-Z0-9_-]{1,64}$. This is deliberately narrower, and every
# narrowing earns its place: the name is also a .py filename, so con.py and
# nul.py are unopenable on Windows; a hyphen is legal to Groq and illegal in a
# Python module name, and these files are meant to be read by a person; a
# case-insensitive filesystem makes GstinCheck and gstincheck one file and two
# entries; and stopping at 47 leaves room for a _v2 suffix.
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,47}$")
RESERVED = {"con", "prn", "aux", "nul", "run", "kwargs",
            *(f"com{n}" for n in range(1, 10)), *(f"lpt{n}" for n in range(1, 10))}


@dataclass(frozen=True, slots=True)
class Tool:
    """One generated tool, and everything known about how it came to exist."""

    name: str
    description: str
    parameters: dict
    source: str
    task_id: str = ""
    family: str = ""
    born_arm: str = ""
    born_round: int = 0
    tests_passed: int = 0
    holdout_passed: int = 0
    holdout_total: int = 0
    digest: str = ""
    written_utc: str = field(default_factory=lambda: datetime.now(timezone.utc)
                             .isoformat(timespec="seconds"))

    def schema(self) -> dict:
        """The Groq function schema this tool is offered as."""
        return {"type": "function",
                "function": {"name": self.name, "description": self.description,
                             "parameters": self.parameters}}


def digest_of(source: str) -> str:
    """A short content hash, used to tell a duplicate from a collision."""
    return "sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]


def name_problem(name: str) -> str:
    """Why this name cannot be used, or an empty string."""
    if not NAME_RE.match(name or ""):
        return (f"'{name}' must match ^[a-z][a-z0-9_]{{2,47}}$ - lowercase letters, "
                "digits and underscores, starting with a letter.")
    if name in RESERVED:
        return f"'{name}' is a reserved name on some systems. Pick another."
    if name in tools.FIXED:
        return (f"'{name}' is already one of your built-in tools. Pick a different "
                "name, or call the existing tool.")
    return ""


class Registry:
    """The tools on offer: three fixed ones, plus whatever has been written.

    The persist switch ablates the READ, not the write. With it off, a new tool
    is still written to disk - into a per-task scratch folder - and still runs;
    it simply is not offered on the next task. Making the write conditional too
    would leave the arms differing in two places at once, and a token gap could
    then be explained away as "it wrote fewer files".
    """

    def __init__(self, *, persist: bool = True, arm: str = "run") -> None:
        self.persist = persist
        self.arm = arm
        self._generated: dict[str, Tool] = {}
        self._task_id = ""

    # --- reading ---------------------------------------------------------
    def load(self) -> int:
        """Read generated/ from disk. Reads text; never executes anything."""
        if not self.persist:
            return 0
        for sidecar in sorted(config.GENERATED_DIR.glob("*.json")):
            module = sidecar.with_suffix(".py")
            if not module.exists():
                continue
            spec = json.loads(sidecar.read_text(encoding="utf-8"))
            source = module.read_text(encoding="utf-8")
            if spec.get("digest") and spec["digest"] != digest_of(source):
                # Somebody edited a generated tool by hand. Including it would
                # mean the run contains code no check ever saw.
                print(f"  skipping {module.name}: its source no longer matches its record")
                continue
            spec.pop("written_utc", None)
            self._generated[spec["name"]] = Tool(**{**spec, "source": source})
        return len(self._generated)

    def schemas(self) -> list[dict]:
        """Every tool on offer. Fixed first, then generated in the order written.

        The order is fixed rather than incidental: an array ordered by dict
        insertion changes the prompt bytes, and therefore the cache key, for
        reasons that have nothing to do with the mechanism being measured.
        """
        return [*tools.SCHEMAS, *(tool.schema() for tool in self._generated.values())]

    def names(self) -> list[str]:
        return [*tools.FIXED, *self._generated]

    def generated(self) -> list[Tool]:
        return list(self._generated.values())

    def get(self, name: str) -> Tool | None:
        return self._generated.get(name)

    def stamp(self) -> list[dict]:
        """What was on offer, for the results row."""
        return [{"name": t.name, "digest": t.digest, "task_id": t.task_id,
                 "family": t.family} for t in self._generated.values()]

    # --- writing ---------------------------------------------------------
    def begin_task(self, task_id: str) -> None:
        """Start a task. With persist off, the toolbox goes back to the fixed three."""
        self._task_id = task_id
        if not self.persist:
            self._generated.clear()

    def collision(self, name: str, source: str) -> str:
        """'', 'duplicate', or a rejection reason for an existing name."""
        existing = self._generated.get(name)
        if existing is None:
            return ""
        if existing.digest == digest_of(source):
            return "duplicate"
        return (f"a tool called '{name}' already exists and its code is different. "
                "It is in your tool list - call it, or write a new tool under "
                "another name.")

    def add(self, tool: Tool) -> Path:
        """Register a tool and write it to disk. Returns the file it landed in."""
        folder = (config.GENERATED_DIR if self.persist else
                  config.GENERATED_DIR / "_scratch" / self.arm / (self._task_id or "task"))
        folder.mkdir(parents=True, exist_ok=True)
        module = folder / f"{tool.name}.py"
        module.write_text(tool.source, encoding="utf-8")
        spec = {k: v for k, v in asdict(tool).items() if k != "source"}
        module.with_suffix(".json").write_text(
            json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self._generated[tool.name] = tool
        return module

    # --- calling ---------------------------------------------------------
    def dispatch(self, name: str, args: dict) -> str:
        """Run one tool and return the text the model sees. Never raises."""
        if name in tools.FIXED:
            return tools.call(name, args)
        tool = self._generated.get(name)
        if tool is None:
            return (f"ERROR: there is no tool called {name!r}. You have: "
                    f"{', '.join(self.names())}.")
        results, raw = sandbox.execute(tool.source, [args])
        row = results[0]
        if not row.get("ok"):
            return f"ERROR: {name} failed: {row.get('error', raw.outcome)}"
        text = json.dumps(row["value"], ensure_ascii=False, default=str)
        return text[:config.MAX_RESULT_CHARS] + (
            "\n... [truncated]" if len(text) > config.MAX_RESULT_CHARS else "")


def used_generated(name: str) -> bool:
    """Whether a tool name refers to something the agent wrote."""
    return name not in tools.FIXED


def clear_generated(*, scratch: bool = True) -> int:
    """Empty the generated folder so a run starts from three tools.

    Only the persist arm writes here, so there is nothing to collide with; what
    this prevents is a fresh sweep inheriting yesterday's toolbox and scoring a
    reuse it never earned.
    """
    import shutil

    removed = 0
    for path in config.GENERATED_DIR.glob("*.py"):
        path.unlink()
        path.with_suffix(".json").unlink(missing_ok=True)
        removed += 1
    if scratch:
        shutil.rmtree(config.GENERATED_DIR / "_scratch", ignore_errors=True)
    return removed


if __name__ == "__main__":
    import shutil

    assert name_problem("gstin_check") == ""
    assert "must match" in name_problem("GstinCheck")
    assert "must match" in name_problem("a")
    assert "must match" in name_problem("check-digit")
    assert "reserved" in name_problem("nul")
    assert "built-in" in name_problem("calc")

    source = ("def run(**kwargs):\n    'Double a number.'\n    return kwargs['n'] * 2\n")
    tool = Tool(name="doubler", description="Double a number.",
                parameters={"type": "object", "additionalProperties": False,
                            "required": ["n"], "properties": {"n": {"type": "integer"}}},
                source=source, task_id="t1", family="demo", born_arm="selftest",
                digest=digest_of(source))

    scratch = config.GENERATED_DIR / "_scratch" / "selftest"
    shutil.rmtree(scratch, ignore_errors=True)
    off = Registry(persist=False, arm="selftest")
    off.begin_task("t1")
    written = off.add(tool)
    assert written.exists() and "_scratch" in str(written), written
    assert off.dispatch("doubler", {"n": 21}) == "42"
    assert len(off.schemas()) == len(tools.SCHEMAS) + 1
    off.begin_task("t2")
    assert off.get("doubler") is None, "persist=False must not carry a tool forward"
    assert len(off.schemas()) == len(tools.SCHEMAS)
    print("persist off  -> the tool was written, ran, and did not survive the task")

    on = Registry(persist=True, arm="selftest")
    on.begin_task("t1")
    on.add(tool)
    on.begin_task("t2")
    assert on.get("doubler") is not None, "persist=True must carry it forward"
    assert on.collision("doubler", source) == "duplicate"
    assert "already exists" in on.collision("doubler", source + "\n# changed\n")
    print("persist on   -> it survived, and an identical rewrite is a duplicate")

    assert "ERROR" in on.dispatch("missing_tool", {})
    broken = Tool(name="broken", description="Fail.", parameters={},
                  source="def run(**kwargs):\n    'Fail.'\n    return 1 / 0\n")
    on._generated["broken"] = broken
    assert on.dispatch("broken", {}).startswith("ERROR: broken failed:")
    print(f"a crash      -> {on.dispatch('broken', {})[:60]}")

    shutil.rmtree(scratch, ignore_errors=True)
    for leftover in config.GENERATED_DIR.glob("doubler.*"):
        leftover.unlink()
    print("OK - registry")
