"""The script a generated tool actually becomes, and how its results are read.

Inputs:  one generated module's source, and a batch of calls
Outputs: the text of a runnable script, and the parsed results of running it

The script is three parts in one file: a preamble that locks the interpreter
down, the model's source byte for byte, and a footer that reads calls from
stdin and prints one result line per call.

Two details here are not style, they are bugs already paid for.

The sentinel is plain ASCII. str.splitlines() splits on \\x1c, \\x1d, \\x1e,
\\x0b, \\x0c, \\x85, \\u2028 and \\u2029 as well as newline, so a sentinel built
from control characters gets torn in half by the parent that reads it.

Every name this file introduces starts with _TS_, and the guard refuses source
that binds one. Otherwise a helper named `json` in the model's module would
shadow the footer's own serialiser, and the footer would break on a tool that
is merely badly named rather than hostile.
"""

from __future__ import annotations

import json

from src import config

SENTINEL = "@@TOOLSMITH@@"

# Audit events that end a call. Transcribed from the 3.12 audit-events table.
#
# What is NOT here matters as much. `import` is absent because a cached import
# fires no event at all - on Windows `nt` is in sys.modules before the first
# line of the preamble runs, and nt.system IS os.system - and because
# Counter.most_common() imports heapq the first time it is called, so denying
# imports breaks ordinary tools at random. `exec` and `compile` are absent
# because loading a stdlib module fires both. `marshal.loads` is absent because
# whether it fires depends on whether a .pyc happens to exist, and a control
# that changes between two runs is not a control.
#
# Those four are the static guard's job, where a rejection costs a repair turn
# instead of a wrong answer. This layer blocks what an import is FOR.
_DENY = """frozenset({
        "os.system", "os.exec", "os.spawn", "os.posix_spawn", "os.fork",
        "os.forkpty", "os.startfile", "subprocess.Popen", "os.kill", "os.killpg",
        "socket.__new__", "socket.connect", "socket.bind", "socket.gethostbyname",
        "socket.getaddrinfo", "socket.sendto", "urllib.Request", "webbrowser.open",
        "ctypes.dlopen", "ctypes.dlsym", "ctypes.dlsym/handle", "ctypes.call_function",
        "os.remove", "os.rename", "os.mkdir", "os.rmdir", "os.link", "os.symlink",
        "os.chmod", "os.chown", "os.truncate", "os.listdir", "os.scandir", "os.walk",
        "glob.glob", "glob.glob/2", "shutil.copyfile", "shutil.rmtree", "shutil.move",
        "os.putenv", "os.unsetenv", "os.add_dll_directory",
        "winreg.OpenKey", "winreg.QueryValue", "winreg.EnumKey",
        "pickle.find_class", "builtins.input", "builtins.breakpoint",
        "sys.settrace", "sys.setprofile",
        "cpython.run_file", "cpython.run_module", "cpython.run_command",
    })"""

_IMPORT_LINES = "\n".join(f"import {name}" for name in sorted(config.ALLOWED_IMPORTS))

PREAMBLE = f'''"""Generated tool, wrapped for execution. Do not edit by hand."""
import sys as _TS_sys
import json as _TS_json
import traceback as _TS_traceback
{_IMPORT_LINES}
import heapq, numbers, operator, locale

# Deep recursion raises RecursionError instead of taking the C stack with it.
# int -> str already caps at 4300 digits by default in 3.12, so a tool asking
# for str(10**10**7) fails fast without any help from here.
_TS_sys.setrecursionlimit(2000)


class _TS_Violation(BaseException):
    """Raised by the audit hook. Deliberately not an Exception.

    A tool that wraps its payload in `try: ... except Exception: pass` would
    otherwise swallow the block and return a clean-looking result. The
    operation still would not have happened, but the red-team report would say
    it was never attempted, which is worse than a failure. Only a bare
    `except:` or `except BaseException:` can catch this - and the guard refuses
    both, which is the half of this that makes it work.
    """


def _TS_install() -> None:
    """Install the audit hook, closed over its policy.

    roots and deny are locals, not module globals, so nothing the tool defines
    later can rebind them. A tool that set _TS_ROOTS = ("",) would otherwise
    make every path start with an allowed prefix and open the filesystem back
    up. Once added, an audit hook cannot be removed: there is no API for it.
    """
    roots = tuple(path for path in _TS_sys.path if path) + (__file__,)
    deny = {_DENY}

    def _hook(event, args):
        if event in deny:
            raise _TS_Violation("sandbox denied: " + event)
        if event == "open":
            path, mode = args[0], args[1] or ""
            if not isinstance(path, str):
                raise _TS_Violation("sandbox denied: open by file descriptor")
            if any(flag in mode for flag in "wxa+"):
                raise _TS_Violation("sandbox denied: writing " + str(path))
            # Reading is confined to the standard library and this script.
            # That is not a loophole, it is how import works: a lazy stdlib
            # import at any later point still has to open a file. __file__ is
            # here because traceback.format_exc() opens the script to render
            # the offending line, and without it every error inside the
            # sandbox turns into a sandbox violation about the sandbox.
            if not path.startswith(roots):
                raise _TS_Violation("sandbox denied: reading " + path)

    _TS_sys.addaudithook(_hook)


_TS_install()
del _TS_install

# --- the model's source begins here -----------------------------------------
'''

FOOTER = f'''
# --- the model's source ends here -------------------------------------------
# Re-imported under private names AFTER the tool's source, so a helper that
# happens to be called `json` cannot break the serialiser.
import json as _TS_json_out
import sys as _TS_sys_out
import traceback as _TS_tb_out

for _TS_line in _TS_sys_out.stdin.read().split("\\n"):
    if not _TS_line.strip():
        continue
    try:
        _TS_result = {{"ok": True, "value": run(**_TS_json_out.loads(_TS_line))}}
        # Prove it serialises before anything is printed. A tool returning a
        # set would otherwise raise after the try block, print nothing, and be
        # indistinguishable from a timeout.
        _TS_json_out.dumps(_TS_result)
    except BaseException as _TS_exc:
        # BaseException, so sys.exit() inside run() is one failed call rather
        # than the end of the batch.
        _TS_result = {{"ok": False,
                       "error": type(_TS_exc).__name__ + ": " + str(_TS_exc),
                       "trace": _TS_tb_out.format_exc()[-1200:]}}
    print("{SENTINEL}" + _TS_json_out.dumps(_TS_result, default=str, ensure_ascii=False))
'''

RESERVED_PREFIX = "_TS_"


def build_script(source: str) -> str:
    """The full script: lockdown, the model's source verbatim, then the footer.

    The model's source is not rewritten. One useful side effect of sitting
    below the preamble is that an encoding cookie in the model's first two
    lines is no longer in the file's first two lines, so it does nothing.
    """
    return PREAMBLE + source.rstrip() + "\n" + FOOTER


def build_calls(calls: list[dict]) -> str:
    """One JSON object per line, which is what the footer reads from stdin."""
    return "\n".join(json.dumps(call, ensure_ascii=False) for call in calls) + "\n"


def parse_lines(stdout: str, expected: int) -> tuple[list[dict], str]:
    """(results, whatever the tool printed itself).

    Split on "\\n" rather than splitlines(), because splitlines() also breaks
    on half a dozen control characters and a tool that prints one would have
    its result line cut in two. Any line without the sentinel is the tool's own
    output and is kept for display, never parsed.
    """
    results: list[dict] = []
    stray: list[str] = []
    for line in stdout.split("\n"):
        line = line.rstrip("\r")
        if line.startswith(SENTINEL):
            try:
                results.append(json.loads(line[len(SENTINEL):]))
            except json.JSONDecodeError as exc:
                results.append({"ok": False, "error": f"unreadable result line: {exc}"})
        elif line.strip():
            stray.append(line)
    while len(results) < expected:
        results.append({"ok": False, "error": (
            f"no result for call {len(results) + 1}; the process stopped after "
            f"{len(results)} of {expected} calls")})
    return results[:expected], "\n".join(stray)


if __name__ == "__main__":
    script = build_script("def run(**kwargs):\n    return kwargs\n")
    assert "_TS_install()" in script and SENTINEL in script
    assert script.index("_TS_install()") < script.index("def run")
    assert script.index("def run") < script.index("_TS_json_out")
    assert all(ch.isprintable() and ch.isascii() for ch in SENTINEL)

    good, stray = parse_lines(
        f'debug line\n{SENTINEL}{{"ok": true, "value": 7}}\r\n', 1)
    assert good == [{"ok": True, "value": 7}] and stray == "debug line", (good, stray)

    short, _ = parse_lines(f'{SENTINEL}{{"ok": true, "value": 1}}\n', 3)
    assert len(short) == 3 and short[1]["ok"] is False
    assert "stopped after 1 of 3" in short[1]["error"], short[1]

    # A tool that prints a control character must not split its own result.
    torn, _ = parse_lines(f'{SENTINEL}{{"ok": true, "value": "a\\u001eb"}}\n', 1)
    assert torn[0]["value"] == "ab", torn

    assert build_calls([{"n": 1}, {"n": 2}]).count("\n") == 2
    print(f"sentinel {SENTINEL} | preamble {len(PREAMBLE)} chars | "
          f"footer {len(FOOTER)} chars")
    print("OK - protocol")
