"""The RAG pipeline under test, with every stage wrapped in a trace step.

Inputs:  a query and the retrieval policy (top-k, minimum similarity)
Outputs: a saved Trace holding steps, per-passage rows, a diagnosis and the answer

Five steps: retrieve, cut, judge, answer, diagnose. Only `cut` decides what
the customer's answer is built from. The judge reads the wider pool on the
side and changes nothing, so the trace shows the pipeline exactly as it runs.
"""

from src import config
from src.diagnose import build_rows, diagnose, metrics, retrieval_decision
from src.judge import answer, judge_pool
from src.retriever import search
from src.tracer import Trace


async def run(query: str, top_k: int = config.TOP_K, min_sim: float = config.MIN_SIMILARITY) -> Trace:
    """Run one query end to end and return its trace (already saved to disk)."""
    pool_size = max(config.JUDGE_POOL, top_k)  # the judge must see at least everything the pipeline keeps
    trace = Trace(query=query, settings={
        "top_k": top_k, "min_similarity": min_sim, "judge_pool": pool_size,
        "embedder": "chroma default (all-MiniLM-L6-v2, ONNX)", "judge_model": config.JUDGE_MODEL,
    })

    with trace.step("retrieve") as s:
        pool = search(query, pool_size)
        s.summary = {"returned": len(pool), "best_similarity": pool[0].similarity}

    with trace.step("cut") as s:
        kept = [c for c in pool if retrieval_decision(c, top_k, min_sim)[0]]
        s.summary = {"kept": len(kept), "dropped": len(pool) - len(kept)}

    with trace.step("judge") as s:
        judged = await judge_pool(query, pool)
        s.summary = {"judged": f"{len(judged.verdicts)}/{len(pool)}", "tokens": judged.tokens}
        if judged.error:
            s.summary["error"] = judged.error

    with trace.step("answer") as s:
        try:
            trace.answer, s.summary["tokens"] = await answer(query, kept)
        except Exception as exc:  # a failed answer should not hide the retrieval trace
            trace.answer = f"(answer call failed: {type(exc).__name__})"
            s.summary["error"] = str(exc)[:300]

    with trace.step("diagnose") as s:
        trace.rows = build_rows(pool, judged, top_k, min_sim)
        trace.diagnosis = {**diagnose(trace.rows, len(pool), top_k, min_sim), "metrics": metrics(trace.rows)}
        s.summary = {"headline": trace.diagnosis["headline"]}

    trace.save()
    return trace
