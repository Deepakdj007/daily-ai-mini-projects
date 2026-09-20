"""Running a generated tool in a stripped-down interpreter.

Inputs:  a generated module's source, and a batch of calls
Outputs: one result per call, and how the process ended

THE DIVISION OF LABOUR, which is the idea worth taking from this file.

The static guard reads source, so it sees names, not values and not time. It
cannot know that `while True: pass` never ends, or how many bytes a loop will
print. It is coarse on purpose: `import os` is refused on the name alone,
without asking what the tool meant to do with it, because asking would mean
evaluating. Every rejection is a proposal the model can act on, which is why
the guard is the layer that speaks.

This layer sees resolved operations - the real path being opened, the real
command being spawned - which is exactly what the guard cannot resolve. But by
then the model is gone: there is no repair turn, only a dead process. And it
cannot see anything that never becomes an operation, so
`().__class__.__mro__[-1].__subclasses__()` raises no audit event at all.

The overlap is the point. `import os; os.system(...)` is refused twice, on the
name and on the os.system event, and the two refusals are independent.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from src import config, protocol

_CREATE_NO_WINDOW = 0x08000000

# -I  no PYTHON* variables, no script directory on sys.path, no user site. This
#     is what stops a generated tool doing `import config` through an inherited
#     PYTHONPATH and reading the API key out of os.getenv.
# -S  site.py never runs, so a .pth file cannot execute code before our
#     preamble does. The standard library still imports.
# -B  no __pycache__, so the open rule can deny every write mode with no
#     exception carved out for .pyc files.
# -X utf8=1  without it a tool returning a rupee sign dies with a cp1252
#     UnicodeEncodeError the moment stdout is a file. PYTHONIOENCODING would
#     not survive -I; an -X flag does.
_FLAGS = ("-I", "-S", "-B", "-X", "utf8=1")


@dataclass(frozen=True, slots=True)
class Result:
    """One batch of calls, run once."""

    ok: bool
    stdout: str = ""
    stderr: str = ""
    outcome: str = "ok"          # ok | timeout | output | crash
    seconds: float = 0.0


def _child_env(tmp: Path) -> dict[str, str]:
    """The child's whole environment. Nothing is inherited.

    GROQ_API_KEY is the asset being protected, and there is no audit event for
    reading os.environ, so the only way to stop a tool exfiltrating the key is
    for the key not to be in the process. An entirely empty environment starts
    python fine on Windows, so this is a deliberately short list rather than a
    minimum viable one.
    """
    env = {"TEMP": str(tmp), "TMP": str(tmp), "TMPDIR": str(tmp)}
    if os.name == "nt":
        # PATH and COMSPEC are deliberately absent: the child execs nothing.
        env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")
    return env


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill the child and everything it spawned.

    proc.kill() ends the direct child and a grandchild survives holding the
    inherited handles - which is also why stdout goes to a FILE below rather
    than a pipe: waiting on a pipe after killing the child blocks until every
    holder of the write end closes it, so one sleeping grandchild hangs the
    sandbox for as long as it likes.
    """
    if os.name == "nt":
        with suppress(OSError):
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, creationflags=_CREATE_NO_WINDOW)
    else:
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    with suppress(OSError):
        proc.kill()


def run(source: str, calls: list[dict]) -> Result:
    """Run one tool against a batch of calls, in one interpreter start.

    What this does NOT stop, plainly: a spin loop burns a core for the length
    of the timeout; there is no memory cap, because setrlimit does not exist on
    Windows and a Job Object is the only real answer there; the output cap is
    polled, so it overshoots; and this is one OS user with your privileges.
    """
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="toolsmith-") as name:
        tmp = Path(name)
        script, stdin_path = tmp / "tool.py", tmp / "calls.jsonl"
        out_path, err_path = tmp / "out.txt", tmp / "err.txt"
        script.write_text(protocol.build_script(source), encoding="utf-8")
        stdin_path.write_text(protocol.build_calls(calls), encoding="utf-8")

        outcome = "ok"
        with (stdin_path.open("rb") as stdin, out_path.open("wb") as out,
              err_path.open("wb") as err):
            proc = subprocess.Popen(
                [sys.executable, *_FLAGS, str(script)],
                stdin=stdin, stdout=out, stderr=err,
                cwd=str(tmp), env=_child_env(tmp),
                creationflags=_CREATE_NO_WINDOW if os.name == "nt" else 0,
                start_new_session=os.name != "nt")
            deadline = started + config.SANDBOX_TIMEOUT_S
            while proc.poll() is None:
                if time.monotonic() > deadline:
                    outcome = "timeout"
                    break
                if out_path.stat().st_size > config.SANDBOX_MAX_OUTPUT:
                    outcome = "output"
                    break
                time.sleep(config.SANDBOX_POLL_S)
            if outcome != "ok":
                _kill_tree(proc)
            proc.wait()

        stdout = out_path.read_bytes()[:config.SANDBOX_MAX_OUTPUT].decode("utf-8", "replace")
        stderr = err_path.read_bytes()[:8192].decode("utf-8", "replace")

    code = proc.returncode or 0
    if outcome == "ok" and code != 0:
        outcome = "crash"
    return Result(ok=outcome == "ok" and code == 0, stdout=stdout, stderr=stderr,
                  outcome=outcome, seconds=round(time.monotonic() - started, 2))


def execute(source: str, calls: list[dict]) -> tuple[list[dict], Result]:
    """Run a batch and parse it: one result per call, plus the raw run."""
    result = run(source, calls)
    if result.outcome in ("timeout", "output"):
        why = ("did not finish in "
               f"{config.SANDBOX_TIMEOUT_S:.0f}s" if result.outcome == "timeout"
               else "printed more output than the sandbox allows")
        parsed, _ = protocol.parse_lines(result.stdout, 0)
        parsed += [{"ok": False, "error": f"the tool {why}"}] * (len(calls) - len(parsed))
        return parsed[:len(calls)], result
    parsed, _ = protocol.parse_lines(result.stdout, len(calls))
    return parsed, result


if __name__ == "__main__":
    ok, raw = execute("def run(**kwargs):\n    print('debugging')\n"
                      "    return kwargs['n'] * 2\n", [{"n": 3}, {"n": 4}])
    assert raw.ok and [r["value"] for r in ok] == [6, 8], (ok, raw)
    print(f"batch of 2   -> {[r['value'] for r in ok]} in one process, {raw.seconds}s")

    leak, _ = execute("import os\ndef run(**kwargs):\n"
                      "    return os.environ.get('GROQ_API_KEY')\n", [{}])
    assert leak[0]["value"] is None, leak
    print("api key      -> None (the child has no environment)")

    swallow, _ = execute("import os\ndef run(**kwargs):\n    try:\n"
                         "        os.system('echo pwned')\n    except Exception:\n"
                         "        pass\n    return 'looks fine'\n", [{}])
    assert swallow[0]["ok"] is False, swallow
    print("os.system    -> blocked, and `except Exception` did not swallow it")

    wrote, _ = execute("def run(**kwargs):\n"
                       "    open('escaped.txt', 'w').write('x')\n    return 1\n", [{}])
    assert wrote[0]["ok"] is False and "denied" in wrote[0]["error"], wrote
    print(f"open(w)      -> {wrote[0]['error']}")

    spin = run("def run(**kwargs):\n    while True:\n        pass\n", [{}])
    assert spin.outcome == "timeout", spin
    print(f"while True   -> {spin.outcome} after {spin.seconds}s")

    lazy, _ = execute("import collections\ndef run(**kwargs):\n"
                      "    return collections.Counter('aab').most_common(1)\n", [{}])
    assert lazy[0]["ok"], lazy
    print(f"lazy import  -> {lazy[0]['value']} (stdlib reads still work)")

    rupee, _ = execute("def run(**kwargs):\n    return '\u20b91,23,456'\n", [{}])
    assert rupee[0]["value"] == "\u20b91,23,456", rupee
    print(f"unicode      -> {rupee[0]['value']}")
    print("OK - sandbox")
