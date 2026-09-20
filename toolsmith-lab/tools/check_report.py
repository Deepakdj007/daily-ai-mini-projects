"""Exercise the criterion and the gates against results that do not exist yet.

    PYTHONPATH=. uv run python -m tools.check_report

Inputs:  nothing but the corpus and src/report.py
Outputs: pass or fail, and a non-zero exit code

The point is to commit the criterion and the gates while the numbers they will
judge do not exist. A criterion written after seeing a leaderboard is a
threshold chosen to fit the result, and from the outside there is no way to
tell the difference afterwards.

Two things are proved here. That the criterion separates a real effect from a
small one - a registry that saves half the tokens passes, one that saves a tenth
does not. And that every gate can actually fail, because a gate that cannot
fail is decoration with a green tick next to it.
"""

from __future__ import annotations

from src import corpus, report

FIXED_TOOLS = ["calc", "today", "text_stats", "write_tool", "submit_answer"]


def row(task: dict, arm: str, *, correct: bool, completion: int, reused: bool = False,
        rejections: dict | None = None, offered: list[str] | None = None,
        outcome: str = "", retried: int = 0) -> dict:
    """One synthetic results row, shaped exactly like a real one."""
    return {
        "task_id": task["id"], "family": task["family"], "member": task["member"],
        "arm": arm, "answer": "x", "gold": task["answer"], "correct": correct,
        "outcome": outcome or ("correct" if correct else "wrong"),
        "rounds": 3, "llm_calls": 3, "tool_calls": 1,
        "prompt_tokens": completion * 3, "completion_tokens": completion,
        "writes": 1, "registered": 1, "repairs": 0, "rejections": rejections or {},
        "notes": [], "offered": [offered or FIXED_TOOLS], "called": ["gstin_check"],
        "reused_from": "gstin_1" if reused else "", "reused": reused,
        "answering_tool": "gstin_check", "registry": [], "stopped": "",
        "retried": retried,
    }


def build(*, reuse: float = 0.9, ratio: float = 0.45, gap: int = 0,
          first_gap: int = 0, inline_ratio: float = 1.6,
          null_correct: bool = False, fixed_rate: float = 0.2,
          holdout_rejections: int = 3, leak: bool = False) -> dict:
    """A whole synthetic run, with every knob the criterion reads."""
    tasks = corpus.tasks()
    # Every arm walks the tasks in the corpus order, because one of the gates
    # checks exactly that - and a builder that emits them in a different order
    # fails its own gate for a reason that has nothing to do with the run.
    repeats = [t["id"] for t in tasks if t["member"] > 1]
    firsts = [t["id"] for t in tasks if t["member"] == 1]
    rows: list[dict] = []

    for index, task in enumerate(tasks):
        if task["member"] > 1:
            rank = repeats.index(task["id"]) / max(len(repeats) - 1, 1)
            rows.append(row(task, "holdout", correct=True, completion=1000))
            rows.append(row(task, "persist", completion=int(1000 * ratio),
                            correct=rank >= (-gap / 100.0), reused=rank < reuse))
            rows.append(row(task, "inline", correct=True,
                            completion=int(1000 * inline_ratio)))
        else:
            rank = firsts.index(task["id"]) / len(firsts)
            rows.append(row(task, "holdout", correct=True, completion=1400))
            rows.append(row(task, "persist", completion=1400,
                            correct=rank >= (first_gap / 100.0)))
            rows.append(row(task, "inline", correct=True, completion=1400))
        rows.append(row(task, "none", correct=null_correct, completion=200))
        # Spread across families rather than taking the first N tasks in order:
        # the first N are all one family, which trips the gate that watches for
        # a family the fixed toolset can solve outright.
        rows.append(row(task, "fixed", completion=400,
                        correct=((index * 11) % len(tasks)) / len(tasks) < fixed_rate))
        rows.append(row(task, "tested", correct=True, completion=1100))

    for index in range(holdout_rejections):
        target = [r for r in rows if r["arm"] == "holdout"][index * 5]
        target["rejections"] = {"holdout": 1}
    if leak:
        [r for r in rows if r["arm"] == "tested"][0]["offered"] = [
            ["calc", "gstin_check", "submit_answer"]]

    return {"manifest": {"model": "synthetic", "profile": "full", **corpus.stamp(),
                         "prompt_version": "1"},
            "results": rows}


def named(payload: dict) -> dict[str, bool]:
    return {name: ok for name, ok, _ in report.gates(payload)}


def main() -> int:
    failures: list[str] = []

    def expect(label: str, condition: bool) -> None:
        print(f"  {'PASS' if condition else 'FAIL'}  {label}")
        if not condition:
            failures.append(label)

    print("the criterion separates a real effect from a small one")
    met, summary, _ = report.verdict(build())
    expect(f"a registry that halves the cost and reuses 90% is MET  [{summary[:60]}]", met)
    expect("a registry that saves a tenth is NOT MET",
           not report.verdict(build(ratio=0.9))[0])
    expect("a registry that rarely reuses is NOT MET",
           not report.verdict(build(reuse=0.3))[0])
    expect("a registry that is 20 points less correct is NOT MET",
           not report.verdict(build(gap=-20))[0])
    expect("a control cheaper than the registry is NOT MET",
           not report.verdict(build(inline_ratio=0.4))[0])
    expect("a gap on first members, where nothing can be reused, is NOT MET",
           not report.verdict(build(first_gap=20))[0])

    print("\nand every gate can fail")
    clean = named(build())
    expect(f"a clean run passes every gate "
           f"{[n for n, ok in clean.items() if not ok] or ''}", all(clean.values()))
    expect("a null arm that answers everything voids the run",
           not named(build(null_correct=True))["the null arm cannot guess"])
    expect("a fixed toolset that solves the families voids the run",
           not named(build(fixed_rate=1.0))["the fixed toolset cannot solve a family"])
    expect("a holdout that never fires is called decoration",
           not named(build(holdout_rejections=0))["the held-out check is not decoration"])
    expect("a written tool offered to a non-persist arm voids the run",
           not named(build(leak=True))["the toolbox did not leak between tasks"])

    print(f"\n{'OK - the criterion and the gates both work' if not failures else failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
