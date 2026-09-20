"""The toolbox: what the agent can call, including what it wrote itself.

Inputs:  the generated/ directory, and tools handed over by src/smith.py
Outputs: Groq function schemas, and the result of calling one

A written tool is never imported into this process. Every call goes back
through the sandbox, for three reasons.

The guard is a static pass and the sandbox is the enforcement. Running
model-written code in-process would make every guard bypass a live bypass
against a process holding the API key; in a subprocess a bypass is a crash.

A tool is validated by being run in the sandbox, so if it were later called by
some other mechanism, the thing that passed the checks would not be the thing
answering.

And loading the toolbox must never execute anything. `generated/` is read at
startup, possibly days later; `import generated.gstin_check` would run
module-level code before any check looked at it. Reading text cannot.

The honest price: a subprocess keeps no state between calls, so every written
tool has to be a pure function of its arguments. The system prompt says so.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from src import config, sandbox, tools

# Groq accepts ^[a-zA-Z0-9_-]{1,64}$. This is narrower, and each narrowing
# earns its place: the name is also a .py filename, so con.py and nul.py are
# unopenable on Windows; a hyphen is legal to Groq and illegal in a Python
# module name, and these files are meant to be read by a person; and a
# case-insensitive filesystem makes GstinCheck and gstincheck one file.
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,47}$")
RESERVED = {"con", "prn", "aux", "nul", "run", "kwargs",
            *(f"com{n}" for n in range(1, 10)), *(f"lpt{n}" for n in range(1, 10))}


@dataclass(frozen=True, slots=True)
class Tool:
    """One written tool, and how it came to exist."""

    name: str
    description: str
    parameters: dict
    source: str
    task_id: str = ""
    tests_passed: int = 0
    holdout_passed: int = 0
    digest: str = ""

    def schema(self) -> dict:
        """The Groq function schema this tool is offered as."""
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": self.parameters}}


def digest_of(source: str) -> str:
    """A short content hash, used to tell a duplicate from a collision."""
    return "sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]


def name_problem(name: str) -> str:
    """Why this name cannot be used, or an empty string."""
    if not NAME_RE.match(name or ""):
        return (f"'{name}' must match ^[a-z][a-z0-9_]{{2,47}}$ - lowercase "
                "letters, digits and underscores, starting with a letter.")
    if name in RESERVED:
        return f"'{name}' is a reserved name on some systems. Pick another."
    if name in tools.FIXED:
        return (f"'{name}' is already one of your built-in tools. Pick another "
                "name, or call the existing tool.")
    return ""


class Registry:
    """Two fixed tools, plus whatever the agent has written."""

    def __init__(self) -> None:
        self._written: dict[str, Tool] = {}

    def load(self) -> int:
        """Read generated/ from disk. Reads text; never executes anything."""
        for sidecar in sorted(config.GENERATED_DIR.glob("*.json")):
            module = sidecar.with_suffix(".py")
            if not module.exists():
                continue
            spec = json.loads(sidecar.read_text(encoding="utf-8"))
            source = module.read_text(encoding="utf-8")
            if spec.get("digest") and spec["digest"] != digest_of(source):
                # Somebody edited a written tool by hand. Including it would
                # mean running code no check ever saw.
                print(f"  skipping {module.name}: its source no longer matches its record")
                continue
            self._written[spec["name"]] = Tool(**{**spec, "source": source})
        return len(self._written)

    def schemas(self) -> list[dict]:
        """Every tool on offer: fixed first, then written, in the order written."""
        return [*tools.SCHEMAS, *(t.schema() for t in self._written.values())]

    def names(self) -> list[str]:
        return [*tools.FIXED, *self._written]

    def written(self) -> list[Tool]:
        return list(self._written.values())

    def get(self, name: str) -> Tool | None:
        return self._written.get(name)

    def collision(self, name: str, source: str) -> str:
        """'', 'duplicate', or a rejection reason for an existing name."""
        existing = self._written.get(name)
        if existing is None:
            return ""
        if existing.digest == digest_of(source):
            return "duplicate"
        return (f"a tool called '{name}' already exists and its code is "
                "different. It is in your tool list - call it, or write a new "
                "tool under another name.")

    def add(self, tool: Tool) -> Path:
        """Register a tool and write it to disk as a readable .py file."""
        config.GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        module = config.GENERATED_DIR / f"{tool.name}.py"
        module.write_text(tool.source, encoding="utf-8")
        spec = {k: v for k, v in asdict(tool).items() if k != "source"}
        module.with_suffix(".json").write_text(
            json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self._written[tool.name] = tool
        return module

    def dispatch(self, name: str, args: dict) -> str:
        """Run one tool and return the text the model sees. Never raises."""
        if name in tools.FIXED:
            return tools.call(name, args)
        tool = self._written.get(name)
        if tool is None:
            return (f"ERROR: there is no tool called {name!r}. You have: "
                    f"{', '.join(self.names())}.")
        results, raw = sandbox.execute(tool.source, [args])
        row = results[0]
        if not row.get("ok"):
            return f"ERROR: {name} failed: {row.get('error', raw.outcome)}"
        text = json.dumps(row["value"], ensure_ascii=False, default=str)
        return (text[:config.MAX_RESULT_CHARS]
                + ("\n... [truncated]" if len(text) > config.MAX_RESULT_CHARS else ""))


def clear() -> int:
    """Empty the toolbox so a demo starts from the two fixed tools."""
    removed = 0
    for path in config.GENERATED_DIR.glob("*.py"):
        path.unlink()
        path.with_suffix(".json").unlink(missing_ok=True)
        removed += 1
    shutil.rmtree(config.GENERATED_DIR / "__pycache__", ignore_errors=True)
    return removed


if __name__ == "__main__":
    import tempfile

    # Into a temporary folder, never the real one. This self-test registers a
    # tool and clears up after itself, and an earlier version did that to
    # generated/ - which quietly deleted the tools the agent had actually
    # written. A check that destroys real state is a bug, however green it is.
    config.GENERATED_DIR = Path(tempfile.mkdtemp(prefix="toolsmith-registry-"))

    assert name_problem("gstin_check") == ""
    assert "must match" in name_problem("GstinCheck")
    assert "must match" in name_problem("check-digit")
    assert "reserved" in name_problem("nul")
    assert "built-in" in name_problem("calc")

    source = "def run(**kwargs):\n    return kwargs['n'] * 2\n"
    tool = Tool(name="doubler", description="Double a number.", source=source,
                parameters={"type": "object", "additionalProperties": False,
                            "required": ["n"], "properties": {"n": {"type": "integer"}}},
                task_id="demo", digest=digest_of(source))
    reg = Registry()
    reg.add(tool)
    assert reg.dispatch("doubler", {"n": 21}) == "42"
    assert len(reg.schemas()) == len(tools.SCHEMAS) + 1
    assert reg.collision("doubler", source) == "duplicate"
    assert "already exists" in reg.collision("doubler", source + "\n# changed\n")
    print(f"written tool -> {reg.dispatch('doubler', {'n': 21})} (through the sandbox)")

    fresh = Registry()
    assert fresh.load() >= 1 and fresh.get("doubler") is not None
    print("reload       -> read back from disk without importing anything")
    assert "ERROR" in reg.dispatch("missing", {})
    reg._written["broken"] = Tool(name="broken", description="Fail.", parameters={},
                                  source="def run(**kwargs):\n    return 1 / 0\n")
    print(f"a crash      -> {reg.dispatch('broken', {})[:58]}")
    print(f"cleared {clear()} tool(s)")
    print("OK - registry")
