"""Every check that costs no API tokens. Run it before spending any.

The policy gates, the Send payload, the report table and the two measurement
harnesses are all deterministic, so they can be checked exactly and for free.
Only the model's judgement needs real calls, and that is what `ask-rate` and
`compare` are for.

    PYTHONPATH=. uv run python -m tools.selftest

Exits non-zero if anything fails, so it works as a pre-commit gate.
"""

import tempfile
from pathlib import Path

from langgraph.types import Send

from src import config

# Redirect every file this writes before importing anything that writes one.
config.OUTPUT_DIR = Path(tempfile.mkdtemp())

from src import askrate, compare, report  # noqa: E402
from src.cord import desk_node, suppressed_questions  # noqa: E402
from src.graph import route_after_lead  # noqa: E402

FAILURES: list[str] = []


def check(name: str, got, want) -> None:
    """Assert and record, rather than stopping at the first failure."""
    if got == want:
        print(f"  [PASS] {name}")
        return
    FAILURES.append(name)
    print(f"  [FAIL] {name}\n         got  {got!r}\n         want {want!r}")


def lead_state(policy: str, asked: bool = False, flags: bool = True) -> dict:
    """A lead's plan, as the router would see it."""
    return {
        "brief": "B", "ask_policy": policy, "interpretation": "I",
        "sub_questions": ["q1", "q2"],
        "needs_human": flags, "question": "which one?" if flags else "",
        "clarifications": (
            [{"id": "lead", "asked_by": "research lead", "text": "which one?",
              "answer": "the second"}] if asked else []
        ),
        "findings": [],
    }


STUCK = [
    {"index": 1, "question": "q1", "summary": "s", "needs_human": True,
     "ask": "our internal number?", "sources": [], "failed": "", "n_results": 5},
    {"index": 2, "question": "q2", "summary": "s", "needs_human": False,
     "ask": "", "sources": [], "failed": "", "n_results": 5},
]


def test_router() -> None:
    """Who may stop the line before any work starts."""
    print("\nrouter: the lead's gate")
    check("never + flag dispatches anyway",
          route_after_lead(lead_state("never"))[0].node, "researcher")
    check("lead + flag asks", route_after_lead(lead_state("lead")), "ask_lead")
    check("team + flag asks", route_after_lead(lead_state("team")), "ask_lead")
    check("no flag dispatches",
          isinstance(route_after_lead(lead_state("team", flags=False)), list), True)
    check("already answered never asks twice",
          isinstance(route_after_lead(lead_state("team", asked=True)), list), True)

    print("\nrouter: the Send payload")
    sends = route_after_lead(lead_state("never"))
    check("one Send per sub-question", len(sends), 2)
    check("all are Sends", all(isinstance(s, Send) for s in sends), True)
    check("payload carries every key the researcher reads",
          sorted(sends[0].arg.keys()),
          ["ask_policy", "brief", "clarifications", "index", "interpretation",
           "question"])
    check("indexes are 1-based and distinct", [s.arg["index"] for s in sends], [1, 2])
    check("no sub-questions falls back to the brief",
          route_after_lead({**lead_state("never"), "sub_questions": []})[0].arg["question"],
          "B")


def test_desk() -> None:
    """The researchers' gate, and what a silenced run still reports."""
    print("\ndesk: the researchers' gate")
    for policy in ("never", "lead"):
        check(f"{policy} + stuck researcher does not pause",
              desk_node({"ask_policy": policy, "findings": STUCK}), {})
    check("team + nobody stuck does not pause",
          desk_node({"ask_policy": "team", "findings": [STUCK[1]]}), {})

    print("\nsuppressed questions")
    check("never reports the lead and the researcher",
          [q["id"] for q in suppressed_questions(
              {**lead_state("never"), "findings": STUCK})], ["lead", "r1"])
    check("lead policy reports only the researcher",
          [q["id"] for q in suppressed_questions(
              {**lead_state("lead", asked=True), "findings": STUCK})], ["r1"])
    check("an answered question is never reported as suppressed",
          suppressed_questions(
              {**lead_state("team", asked=True), "findings": [STUCK[1]]}), [])


def test_fixtures() -> None:
    """The guards that stop a rigged or unanswerable comparison."""
    print("\nfixtures")
    import json

    original = config.BRIEFS_PATH.read_text(encoding="utf-8")
    data = json.loads(original)
    check("all briefs load", len(compare.load_briefs(use_all=True)), len(data["briefs"]))

    for label, mutate in (
        ("a token already in the brief is refused",
         lambda b: b.update(brief=b["brief"] + " " + b["tokens"][0])),
        ("an empty oracle_answer is refused",
         lambda b: b.update(oracle_answer="   ")),
    ):
        broken = json.loads(original)
        mutate(broken["briefs"][0])
        config.BRIEFS_PATH.write_text(json.dumps(broken), encoding="utf-8")
        try:
            compare.load_briefs(use_all=True)
            check(label, "no error", "AssertionError")
        except AssertionError:
            check(label, "AssertionError", "AssertionError")
        finally:
            config.BRIEFS_PATH.write_text(original, encoding="utf-8")


def test_report() -> None:
    """The table, and the four warnings that say a row proves nothing."""
    print("\nreport table")
    rows = [
        {"id": "x1", "kind": "ambiguous", "policy": "team", "asked": ["research lead"],
         "wanted": [], "turns": 1, "questions": 1, "tokens_found": 2, "tokens_total": 2,
         "failures": [], "dry": 0, "seconds": 9.0},
        {"id": "x2", "kind": "clear", "policy": "team", "asked": ["researcher 1"],
         "wanted": [], "turns": 1, "questions": 1, "tokens_found": 0, "tokens_total": 0,
         "failures": ["RuntimeError: boom"], "dry": 1, "seconds": 3.0},
        {"id": "x3", "kind": "ambiguous", "policy": "never", "asked": [], "wanted": [],
         "turns": 0, "questions": 0, "tokens_found": 2, "tokens_total": 2,
         "failures": [], "dry": 0, "seconds": 9.0},
    ]
    check("silenced label", report.who(
        {"asked": [], "wanted": ["research lead"]}), "(silenced: lead)")
    check("no planted tokens shows a dash", report.score(rows[1]), "-")
    check("a crashed researcher is visible", report.researchers_ok(rows[1]), "1/2")
    check("render produces one row per input", report.render(rows).row_count, 3)

    notes = report.warnings(rows)
    check("a failed researcher warns", any("never answered" in n for n in notes), True)
    check("a dry researcher warns",
          any("zero search results" in n for n in notes), True)
    check("a leak under never warns", any("nobody to ask" in n for n in notes), True)
    check("a false alarm warns", any("false alarm" in n for n in notes), True)

    path = report.write_markdown(rows, notes, n_briefs=3)
    text = path.read_text(encoding="utf-8")
    header, separator = text.splitlines()[4], text.splitlines()[5]
    check("header and separator agree on columns",
          header.count("|"), separator.count("|"))
    check("warnings reach the file", "## Warnings" in text, True)


def test_harness_wiring() -> None:
    """run_comparison and ask_rate, with the one expensive call stubbed out."""
    print("\nharness wiring (model calls stubbed)")
    calls: list[tuple[str, str]] = []

    def fake_run_one(brief: dict, policy: str) -> dict:
        calls.append((brief["id"], policy))
        asked = policy == "team" and brief["kind"] == "ambiguous"
        return {
            "id": brief["id"], "kind": brief["kind"], "policy": policy,
            "asked": ["research lead"] if asked else [], "wanted": [],
            "turns": int(asked), "questions": int(asked),
            "tokens_found": len(brief["tokens"]) if asked else 0,
            "tokens_total": len(brief["tokens"]),
            "failures": [], "dry": 0, "seconds": 1.0,
        }

    real_run_one, compare.run_one = compare.run_one, fake_run_one
    try:
        compare.run_comparison(use_all=False)
    finally:
        compare.run_one = real_run_one

    check("every brief ran against every policy", len(calls), 8)
    check("both policies used", sorted({p for _, p in calls}), ["never", "team"])
    rows = [ln for ln in (config.OUTPUT_DIR / "compare.md").read_text(
        encoding="utf-8").splitlines()
        if ln.startswith("| b") and not ln.startswith("| brief")]
    check("one data row per run", len(rows), 8)

    samples: list[str] = []

    def fake_lead_node(state: dict) -> dict:
        samples.append(state["brief"])
        flags = "our" in state["brief"].lower()
        return {"interpretation": "i", "sub_questions": ["a", "b"],
                "needs_human": flags, "question": "which?" if flags else ""}

    real_lead, askrate.lead_node = askrate.lead_node, fake_lead_node
    try:
        askrate.ask_rate(samples=3, use_all=False)
    finally:
        askrate.lead_node = real_lead

    check("sampled every brief the requested number of times", len(samples), 12)
    rate_text = (config.OUTPUT_DIR / "ask-rate.md").read_text(encoding="utf-8")
    check("always-flagging brief reads 100%", "| 3/3 | 100% |" in rate_text, True)
    check("never-flagging brief reads 0%", "| 0/3 | 0% |" in rate_text, True)


def main() -> int:
    """Run every free check and report once at the end."""
    for suite in (test_router, test_desk, test_fixtures, test_report,
                  test_harness_wiring):
        suite()
    print(f"\n{'ALL PASS' if not FAILURES else f'{len(FAILURES)} FAILED: {FAILURES}'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
