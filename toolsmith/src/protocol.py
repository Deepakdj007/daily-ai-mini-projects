"""The script a generated tool becomes, and how its results are read back.

Inputs:  one generated module's source, and a batch of calls
Outputs: the text of a runnable script, and the parsed results

Three parts in one file: a preamble that locks the interpreter down, the
model's source byte for byte, and a footer that reads calls from stdin and
prints one result line per call.

Two details are not style, they are bugs already paid for. The sentinel is
plain ASCII, because str.splitlines() also breaks on \\x1c, \\x1d, \\x1e,
\\x0b, \\x0c and \\x85, so a marker built from control characters gets torn in
half by the parent reading it. And every name here starts with _TS_, which the
guard refuses to let generated source bind - otherwise a helper innocently
named `json` would shadow the footer's own serialiser.
"""

from __future__ import annotations

import json

from src import config

SENTINEL = "@@TOOLSMITH@@"
RESERVED_PREFIX = "_TS_"

# Audit events that end a call. What is NOT here matters as much: `import` is
# absent because a cached import fires no event at all - on Windows `nt` is in
# sys.modules before the preamble runs and nt.system IS os.system - and because
# Counter.most_common() imports heapq the first time it is called, so denying
# imports breaks ordinary tools at random. `exec` and `compile` fire whenever a
# stdlib module loads. Those belong to the static guard, where a rejection
# costs a repair turn instead of a wrong answer. This layer blocks what an
# import is FOR.
_DENY = """frozenset({
        "os.system", "os.exec", "os.spawn", "os.posix_spawn", "os.fork",
        "os.forkpty", "os.startfile", "subprocess.Popen", "os.kill",
        "socket.__new__", "socket.connect", "socket.bind", "socket.getaddrinfo",
        "socket.sendto", "urllib.Request", "webbrowser.open",
        "ctypes.dlopen", "ctypes.dlsym", "ctypes.call_function",
        "os.remove", "os.rename", "os.mkdir", "os.rmdir", "os.link", "os.symlink",
        "os.chmod", "os.truncate", "os.listdir", "os.scandir", "os.walk",
        "glob.glob", "shutil.copyfile", "shutil.rmtree", "shutil.move",
        "os.putenv", "os.unsetenv", "winreg.OpenKey", "pickle.find_class",
        "builtins.input", "builtins.breakpoint", "sys.settrace", "sys.setprofile",
        "cpython.run_file", "cpython.run_module", "cpython.run_command",
    })"""

_IMPORTS = "\n".join(f"import {name}" for name in sorted(config.ALLOWED_IMPORTS))

PREAMBLE = f'''"""Generated tool, wrapped for execution. Do not edit by hand."""
import sys as _TS_sys
{_IMPORTS}
import heapq, numbers, operator, locale

_TS_sys.setrecursionlimit(2000)


class _TS_Violation(BaseException):
    """Raised by the audit hook. Deliberately not an Exception.

    A tool wrapping its payload in `try: ... except Exception: pass` would
    otherwise swallow the block and return a clean-looking result. Only a bare
    `except:` can catch this - and the guard refuses those, which is the half
    that makes it work.
    """


def _TS_install() -> None:
    """Install the audit hook, closed over its policy.

    roots and deny are locals, not globals, so nothing the tool defines later
    can rebind them: a tool setting _TS_ROOTS = ("",) would otherwise make
    every path start with an allowed prefix. Once added, an audit hook cannot
    be removed - there is no API for it.
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
            # Reads are confined to the standard library and this script. That
            # is not a loophole, it is how import works - a lazy stdlib import
            # still has to open a file. __file__ is here because
            # traceback.format_exc() opens the script to show the failing line.
            if not path.startswith(roots):
                raise _TS_Violation("sandbox denied: reading " + path)

    _TS_sys.addaudithook(_hook)


_TS_install()
del _TS_install

# --- the model's source begins here -----------------------------------------
'''

FOOTER = f'''
# --- the model's source ends here -------------------------------------------
# Imported under private names AFTER the tool, so a helper called `json`
# cannot break the serialiser.
import json as _TS_json
import sys as _TS_out
import traceback as _TS_tb

for _TS_line in _TS_out.stdin.read().split("\\n"):
    if not _TS_line.strip():
        continue
    try:
        _TS_result = {{"ok": True, "value": run(**_TS_json.loads(_TS_line))}}
        # Prove it serialises before printing. A tool returning a set would
        # otherwise raise after the try block, print nothing, and look exactly
        # like a timeout.
        _TS_json.dumps(_TS_result)
    except BaseException as _TS_exc:
        # BaseException, so sys.exit() inside run() is one failed call rather
        # than the end of the batch.
        _TS_result = {{"ok": False,
                       "error": type(_TS_exc).__name__ + ": " + str(_TS_exc),
                       "trace": _TS_tb.format_exc()[-800:]}}
    print("{SENTINEL}" + _TS_json.dumps(_TS_result, default=str, ensure_ascii=False))
'''


def build_script(source: str) -> str:
    """Lockdown, the model's source verbatim, then the footer.

    One useful side effect of sitting below the preamble: an encoding cookie in
    the model's first two lines is no longer in the file's first two lines, so
    it does nothing.
    """
    return PREAMBLE + source.rstrip() + "\n" + FOOTER


def build_calls(calls: list[dict]) -> str:
    """One JSON object per line, which is what the footer reads from stdin."""
    return "\n".join(json.dumps(call, ensure_ascii=False) for call in calls) + "\n"


def parse_lines(stdout: str, expected: int) -> tuple[list[dict], str]:
    """(results, whatever the tool printed itself).

    Split on "\\n" rather than splitlines(), and treat any line without the
    sentinel as the tool's own output rather than parsing it - models put
    print() debugging in generated code constantly, and a bare json.loads would
    fail for a tool that is perfectly correct.
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
    assert script.index("_TS_install()") < script.index("def run") < script.index("_TS_json")
    assert all(ch.isprintable() and ch.isascii() for ch in SENTINEL)

    good, stray = parse_lines(f'debug\n{SENTINEL}{{"ok": true, "value": 7}}\r\n', 1)
    assert good[0]["value"] == 7 and stray == "debug", (good, stray)
    short, _ = parse_lines(f'{SENTINEL}{{"ok": true, "value": 1}}\n', 3)
    assert len(short) == 3 and "stopped after 1 of 3" in short[1]["error"]
    torn, _ = parse_lines(f'{SENTINEL}{{"ok": true, "value": "a\\u001eb"}}\n', 1)
    assert torn[0]["value"] == "ab", "a control character must not split a result"
    print(f"sentinel {SENTINEL} | preamble {len(PREAMBLE)} chars")
    print("OK - protocol")
