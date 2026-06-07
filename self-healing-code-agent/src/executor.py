"""Run a snippet of Python in an isolated subprocess and capture the result.

This is the agent's "hands". It writes the candidate code to a temp file, runs
it with the same Python interpreter in a fresh process, and reports back whether
it succeeded along with whatever it printed or the traceback it raised.

Inputs:  a string of Python source code.
Outputs: an ExecResult (ok flag + captured stdout + captured stderr).
"""

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from src.config import EXEC_TIMEOUT


@dataclass
class ExecResult:
    """Outcome of running one code snippet."""

    ok: bool          # True if the process exited 0 with no exception
    stdout: str       # whatever the snippet printed
    stderr: str       # traceback / error text if it failed


def run_code(code: str) -> ExecResult:
    """Execute `code` in a subprocess and return what happened.

    Running in a separate process means a crash, a SystemExit, or an infinite
    loop in the generated code can never take down the agent itself.
    """
    # Write the snippet to a temp .py file the child process can import-free run.
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "candidate.py"
        script.write_text(code, encoding="utf-8")

        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True,
                text=True,
                timeout=EXEC_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(
                ok=False,
                stdout="",
                stderr=(
                    f"Execution timed out after {EXEC_TIMEOUT}s "
                    "(possible infinite loop)."
                ),
            )

    return ExecResult(
        ok=proc.returncode == 0,
        stdout=proc.stdout.strip(),
        stderr=proc.stderr.strip(),
    )
