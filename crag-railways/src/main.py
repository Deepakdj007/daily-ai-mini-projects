"""Command line for the CRAG help desk.

Inputs:  `ask "question"` (add --plain for RAG without correction) or `demo`
Outputs: each graph step as it runs, then the cited answer
"""

import argparse
import asyncio

from src.graph import answer, build_graph
from src.kb import retrieve

DEMO = [
    "What time does Tatkal booking open for AC classes?",
    "How much check-in baggage can I carry on an IndiGo domestic flight?",
    "My train ran 5 hours late and I didn't travel. How do I get my money back and how many days does the refund take?",
]


async def run_plain(question: str) -> None:
    """Plain RAG: hand the top passages straight to the model, no grading."""
    docs = retrieve(question)
    print(f"[retrieve] {', '.join(f'{d.id} ({d.similarity:.2f})' for d in docs)}")
    reply = await answer(question, [f"{d.text} [KB:{d.id}]" for d in docs])
    print(f"[answer]   {reply}")


def show(node: str, update: dict) -> None:
    """Print what one graph node just did."""
    if node == "retrieve":
        print(f"[retrieve] {', '.join(f'{d.id} ({d.similarity:.2f})' for d in update['docs'])}")
    elif node == "evaluate":
        grades = ", ".join(f"{pid}={label}" for pid, label in update["labels"].items())
        print(f"[evaluate] {grades}")
        print(f"[verdict]  {update['verdict']}")
    elif node == "refine":
        print(f"[refine]   kept {len(update['knowledge'])} strip(s)")
        for strip in update["knowledge"]:
            print(f"             {strip}")
    elif node == "web":
        print(f"[web]      query: {update['web_query']} -> {len(update['knowledge'])} result(s)")
    elif node == "generate":
        print(f"[answer]   {update['answer']}")


async def run_crag(question: str) -> None:
    """Corrective RAG: stream the graph and print every step."""
    graph = build_graph()
    async for chunk in graph.astream({"question": question}, stream_mode="updates"):
        for node, update in chunk.items():
            show(node, update)


async def run_demo() -> None:
    """Every demo question twice: plain RAG first, then CRAG."""
    for question in DEMO:
        print(f"\n{'=' * 70}\nQ: {question}\n\n--- plain RAG ---")
        await run_plain(question)
        print("\n--- CRAG ---")
        await run_crag(question)


def main() -> None:
    """Parse the command and run it."""
    parser = argparse.ArgumentParser(description="Corrective RAG railway help desk")
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask", help="answer one question")
    ask.add_argument("question")
    ask.add_argument("--plain", action="store_true", help="skip the evaluator (plain RAG)")
    sub.add_parser("demo", help="run the three demo questions, plain vs CRAG")
    args = parser.parse_args()

    if args.command == "demo":
        asyncio.run(run_demo())
    elif args.plain:
        asyncio.run(run_plain(args.question))
    else:
        asyncio.run(run_crag(args.question))


if __name__ == "__main__":
    main()
