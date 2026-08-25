"""Command line entry point.

    run [--limit N]   answer every item with both tiers and cache the results
    evaluate          score all strategies and write the leaderboard
    plot              draw the deferral curve
    all [--limit N]   run, evaluate, plot
    ask "<query>"     route one question and show the decision live
"""

from __future__ import annotations

import argparse

from src.config import HEADLINE_STRICTNESS, LLM_MODEL, require_keys, slm_label


def cmd_run(args: argparse.Namespace) -> None:
    from src import matrix

    require_keys()
    from src.tiers import preflight

    preflight()
    records = matrix.build(limit=args.limit)
    matrix.save(records)
    print(f"\nanswered {len(records)} items")


def cmd_evaluate(_args: argparse.Namespace) -> None:
    from src.evaluate import build_rows
    from src.matrix import load
    from src.report import render_markdown, render_terminal

    data = build_rows(load())
    render_terminal(data)
    render_markdown(data)


def cmd_plot(_args: argparse.Namespace) -> None:
    from src.evaluate import build_rows
    from src.matrix import load
    from src.plot import draw

    draw(build_rows(load()))


def cmd_all(args: argparse.Namespace) -> None:
    cmd_run(args)
    cmd_evaluate(args)
    cmd_plot(args)


def cmd_ask(args: argparse.Namespace) -> None:
    """Route a single query and narrate the decision."""
    from src.grader import extract_answer
    from src.tiers import call_llm, call_slm, preflight
    from src.verifier import should_escalate
    from src.config import SELF_CONSISTENCY_K

    require_keys()
    preflight()
    query = args.query

    print(f"query    : {query}")
    slm = call_slm(query)
    print(f"\n[1] {slm_label()}")
    print(f"    answer : {extract_answer(slm['answer'])!r}")
    print(f"    cost   : ${slm['cost_usd']:.8f}   {slm['latency_s']:.1f}s")

    samples = [slm["answer"]]
    for index in range(1, SELF_CONSISTENCY_K):
        samples.append(call_slm(query, sample=index)["answer"])

    item = {"numeric": False, "choices": None}
    escalate, reason = should_escalate(item, slm["answer"], samples, HEADLINE_STRICTNESS)

    print(f"\n[2] verifier (strictness {HEADLINE_STRICTNESS}, local, $0)")
    print(f"    samples: {[extract_answer(s) for s in samples]}")
    print(f"    verdict: {'ESCALATE — ' + reason if escalate else 'ACCEPT'}")

    if not escalate:
        print(f"\nfinal    : {extract_answer(slm['answer'])!r}")
        print(f"total    : ${slm['cost_usd']:.8f}  (never touched {LLM_MODEL})")
        return

    llm = call_llm(query)
    print(f"\n[3] {LLM_MODEL}")
    print(f"    answer : {extract_answer(llm['answer'])!r}")
    print(f"    cost   : ${llm['cost_usd']:.8f}   {llm['latency_s']:.1f}s")
    print(f"\nfinal    : {extract_answer(llm['answer'])!r}")
    print(
        f"total    : ${slm['cost_usd'] + llm['cost_usd']:.8f}  "
        "(paid for both tiers — escalation is never free)"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="slm-llm-router")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="answer every item with both tiers")
    run.add_argument("--limit", type=int, default=None)
    run.set_defaults(func=cmd_run)

    sub.add_parser("evaluate", help="score strategies").set_defaults(func=cmd_evaluate)
    sub.add_parser("plot", help="draw the deferral curve").set_defaults(func=cmd_plot)

    every = sub.add_parser("all", help="run, evaluate, plot")
    every.add_argument("--limit", type=int, default=None)
    every.set_defaults(func=cmd_all)

    ask = sub.add_parser("ask", help="route one query")
    ask.add_argument("query")
    ask.set_defaults(func=cmd_ask)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    parsed.func(parsed)
