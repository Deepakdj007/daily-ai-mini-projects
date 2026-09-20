"""Running a generated tool in a stripped-down interpreter, once.

Inputs:  a generated module's source, and a batch of calls
Outputs: a Result carrying what it printed, and how it ended

WHAT THE STATIC GUARD CANNOT KNOW. It reads source, so it sees names, not
values and not time. It cannot know that `while True: pass` never ends, or how
many bytes a loop will print, or that a .pth file in site-packages would have
run before the tool's first line. And it is coarse on purpose: `import os` is
refused on the name alone, without asking what the tool meant to do with it,
because asking would mean evaluating. Every rejection is therefore a proposal
the model can act on, which is why the guard is the layer that speaks.

WHAT THIS LAYER CANNOT EXPLAIN. It sees resolved operations - the real path
being opened, the real command being spawned - which is exactly what the guard
cannot resolve. But by then the model is gone: there is no repair turn, only a
dead process and a traceback. It also cannot see anything that never becomes an
operation. `().__class__.__mro__[-1].__subclasses__()` raises no audit event at
all, and would hand back every class in the interpreter if the guard had not
refused the dunders first.

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
#     is the flag that stops a generated tool doing `import config` through an
#     inherited PYTHONPATH and reading the API key straight out of os.getenv.
# -S  site.py never runs, so a .pth file in site-packages cannot execute code
#     before the preamble does. The standard library still imports: those paths
#     come from the path configuration, not from site.
# -B  no __pycache__ in the temp directory, so the open rule can deny every
#     write mode without carving out an exception for .pyc files.
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
    outcome: str = "ok"     # ok | timeout | output | crash
    returncode: int = 0
    seconds: float = 0.0

    @property
    def timed_out(self) -> bool:
        return self.outcome == "timeout"


def _child_env(tmp: Path) -> dict[str, str]:
    """The child's whole environment. Nothing is inherited.

    GROQ_API_KEY is the asset being protected here, and there is no audit event
    for reading os.environ, so the only way to stop a tool exfiltrating the key
    is for the key not to be in the process. An entirely empty environment
    starts python fine on Windows, so this is not a minimum-viable list - it is
    a deliberately short one. TEMP and friends are set only to point
    tempfile at the directory that is about to be deleted.
    """
    env = {"TEMP": str(tmp), "TMP": str(tmp), "TMPDIR": str(tmp)}
    if os.name == "nt":
        # Some DLL-resolution paths consult SYSTEMROOT, and it grants nothing.
        # PATH and COMSPEC are deliberately absent: the child execs nothing,
        # and their absence is one more brake on anything that tries to.
        env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")
    return env


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill the child and everything it spawned.

    proc.kill() alone is not enough. It ends the direct child, and a grandchild
    survives holding the inherited handles - which is also why stdout goes to a
    file here rather than a pipe: waiting on a pipe after killing the child
    blocks until every holder of the write end closes it, so one sleeping
    grandchild hangs the sandbox for as long as it likes.

    The guard already forbids subprocess, so a grandchild exists only if the
    guard was bypassed, which is the exact case this has to survive.
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


def _read(path: Path, limit: int) -> str:
    """Read back at most `limit` bytes of captured output."""
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        return handle.read(limit).decode("utf-8", errors="replace")


def run(source: str, calls: list[dict]) -> Result:
    """Run one guarded tool against a batch of calls, once.

    The whole batch shares a single interpreter start, so twelve checks cost
    one process rather than twelve.

    What this does NOT stop, plainly:

      CPU         `while True: pass` burns a core for the length of the
                  timeout. Caught, but not cheaply.
      memory      there is no cap. setrlimit does not exist on Windows and a
                  Job Object is the only real answer there. A tool that
                  allocates until the machine swaps will do that.
      disk        the output cap is polled, so it overshoots - a tool writing
                  in a tight loop can put tens of megabytes down inside one
                  poll window. It is there to stop a gigabyte, not to be exact.
      privilege   this is one OS user, with your privileges.
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
                start_new_session=os.name != "nt",
            )
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

        stdout = _read(out_path, config.SANDBOX_MAX_OUTPUT)
        stderr = _read(err_path, 8192)

    code = proc.returncode or 0
    if outcome == "ok" and code != 0:
        outcome = "crash"
    return Result(ok=outcome == "ok" and code == 0, stdout=stdout, stderr=stderr,
                  outcome=outcome, returncode=code,
                  seconds=round(time.monotonic() - started, 3))


def execute(source: str, calls: list[dict]) -> tuple[list[dict], Result]:
    """Run a batch and parse it: one result dict per call, plus the raw run.

    A process that died still returns one entry per call, so a caller never has
    to line results up against inputs itself.
    """
    result = run(source, calls)
    if result.outcome in ("timeout", "output"):
        reason = ("the tool did not finish in "
                  f"{config.SANDBOX_TIMEOUT_S:.0f}s" if result.outcome == "timeout"
                  else "the tool printed more output than the sandbox allows")
        parsed, _ = protocol.parse_lines(result.stdout, 0)
        parsed += [{"ok": False, "error": reason}] * (len(calls) - len(parsed))
        return parsed[:len(calls)], result
    parsed, _ = protocol.parse_lines(result.stdout, len(calls))
    return parsed, result


if __name__ == "__main__":
    ok, raw = execute(
        "def run(**kwargs):\n"
        "    'Double a number.'\n"
        "    print('debugging')\n"
        "    return kwargs['n'] * 2\n",
        [{"n": 3}, {"n": 4}],
    )
    assert raw.ok and [row["value"] for row in ok] == [6, 8], (ok, raw)
    print(f"batch of 2   -> {[row['value'] for row in ok]} in {raw.seconds}s")

    leak, _ = execute(
        "import os\ndef run(**kwargs):\n    'Read the key.'\n"
        "    return os.environ.get('GROQ_API_KEY')\n", [{}])
    assert leak[0]["value"] is None, leak
    print(f"api key      -> {leak[0]['value']} (the child has no environment)")

    swallow, _ = execute(
        "import os\ndef run(**kwargs):\n    'Try to run a command.'\n"
        "    try:\n        os.system('echo pwned')\n    except Exception:\n"
        "        pass\n    return 'looks fine'\n", [{}])
    assert swallow[0]["ok"] is False, swallow
    print(f"os.system    -> blocked, and `except Exception` did not swallow it")

    wrote, _ = execute(
        "def run(**kwargs):\n    'Write a file.'\n"
        "    open('escaped.txt', 'w').write('x')\n    return 'wrote'\n", [{}])
    assert wrote[0]["ok"] is False and "denied" in wrote[0]["error"], wrote
    print(f"open(w)      -> {wrote[0]['error']}")

    spin = run("def run(**kwargs):\n    'Spin.'\n    while True:\n        pass\n", [{}])
    assert spin.timed_out, spin
    print(f"while True   -> {spin.outcome} after {spin.seconds}s")

    unserialisable, _ = execute(
        "def run(**kwargs):\n    'Return a set.'\n    return {1, 2, 3}\n", [{}])
    assert unserialisable[0]["ok"] is False, unserialisable
    print(f"returns set  -> reported, not crashed")

    lazy, _ = execute(
        "import collections\ndef run(**kwargs):\n    'Lazy import of heapq.'\n"
        "    return collections.Counter('aab').most_common(1)\n", [{}])
    assert lazy[0]["ok"] and lazy[0]["value"] == [["a", 2]], lazy
    print(f"lazy import  -> {lazy[0]['value']} (stdlib reads still work)")

    rupee, _ = execute(
        "def run(**kwargs):\n    'Return a rupee sign.'\n    return '\u20b91,23,456'\n", [{}])
    assert rupee[0]["value"] == "\u20b91,23,456", rupee
    print(f"unicode      -> {rupee[0]['value']}")
    print("OK - sandbox")
