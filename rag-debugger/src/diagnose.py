"""Turn retrieval results plus judge verdicts into rows, a diagnosis and metrics.

Inputs:  ranked candidates, the judge's verdicts, the retrieval policy
Outputs: one row per passage (with both "why"s), a headline and context precision/recall

Pure functions only. No model calls, no files, so every rule here can be
checked for free by running this module.
"""

import math

from src.judge import JudgeResult
from src.retriever import Candidate

# (in the prompt?, judge says it answers?) -> cell
CELLS = {(True, True): "HIT", (True, False): "NOISE", (False, True): "MISS", (False, False): "CUT"}


def retrieval_decision(c: Candidate, top_k: int, min_sim: float) -> tuple[bool, str]:
    """Did this passage reach the prompt, and the mechanical reason why or why not."""
    if c.rank > top_k:
        return False, f"rank {c.rank} > top-k {top_k}"
    if c.similarity < min_sim:
        return False, f"sim {c.similarity:.3f} < {min_sim:.2f}"
    return True, f"rank {c.rank}, sim {c.similarity:.2f}"


def build_rows(candidates: list[Candidate], judged: JudgeResult, top_k: int, min_sim: float) -> list[dict]:
    """One row per candidate: what the retriever decided next to what the judge thinks."""
    rows = []
    for c in candidates:
        in_prompt, retriever_why = retrieval_decision(c, top_k, min_sim)
        verdict = judged.verdicts.get(c.pid)
        rows.append({
            "rank": c.rank, "pid": c.pid, "text": c.text, "similarity": c.similarity,
            "in_prompt": in_prompt, "retriever_why": retriever_why,
            "label": verdict.label if verdict else "-",
            "judge_why": verdict.reason if verdict else "not judged",
            "cell": CELLS[(in_prompt, verdict.label == "answers")] if verdict else "UNJUDGED",
        })
    return rows


def metrics(rows: list[dict]) -> dict:
    """Context precision (how much of the prompt is useful) and recall (how much of the answer got in)."""
    count = {cell: sum(r["cell"] == cell for r in rows) for cell in ("HIT", "NOISE", "MISS")}
    in_prompt = count["HIT"] + count["NOISE"]
    relevant = count["HIT"] + count["MISS"]
    return {
        "precision": round(count["HIT"] / in_prompt, 2) if in_prompt else None,
        "recall": round(count["HIT"] / relevant, 2) if relevant else None,
        **{cell.lower(): n for cell, n in count.items()},
    }


def recovery(rows: list[dict], top_k: int, min_sim: float) -> dict:
    """The smallest policy change that lets every missed answer in, and what else comes in with it."""
    misses = [r for r in rows if r["cell"] == "MISS"]
    new_k = max(top_k, max(r["rank"] for r in misses))
    # Round the threshold DOWN, or 0.517 would print as 0.52 and still cut the passage.
    new_min = min(min_sim, math.floor(min(r["similarity"] for r in misses) * 100) / 100)
    extra = sum(1 for r in rows if not r["in_prompt"] and r["cell"] != "MISS"
                and r["rank"] <= new_k and r["similarity"] >= new_min)
    changes = []
    if new_k > top_k:
        changes.append(f"top-k={new_k}")
    if new_min < min_sim:
        changes.append(f"min-sim={new_min:.2f}")
    return {"top_k": new_k, "min_sim": new_min, "change": " and ".join(changes), "extra": extra}


def headline(rows: list[dict], pool: int, top_k: int, min_sim: float) -> dict:
    """Name the failure. Priority: judge down > corpus gap > miss > noisy order > healthy."""
    kept = [r for r in rows if r["in_prompt"]]
    hits = [r for r in rows if r["cell"] == "HIT"]
    misses = [r for r in rows if r["cell"] == "MISS"]

    if all(r["cell"] == "UNJUDGED" for r in rows):
        return {"headline": "JUDGE_UNAVAILABLE", "detail": "the judge call failed, so only the retrieval half is shown"}
    if not hits and not misses:
        # The judge only saw the top of the ranking, so "not in the pool" is all it can honestly say.
        if not kept:
            hint = ("Nothing even cleared min-sim, so the retriever may not understand the question at all "
                    "(a different language, say). Check that before writing new content.")
        else:
            hint = (f"If the content exists, it ranks below {pool}; if it doesn't, add it, "
                    "or have the bot say it doesn't know.")
        return {"headline": "CORPUS_GAP", "detail": f"no passage in the top {pool} answers the question. {hint}"}
    if misses:
        fix = recovery(rows, top_k, min_sim)
        ranks = ", ".join(str(r["rank"]) for r in misses)
        return {"headline": "RETRIEVAL_MISS",
                "detail": f"the answer was retrieved at rank {ranks} and cut. {fix['change']} would reach it, "
                          f"but also lets in {fix['extra']} more passage(s) that don't answer.",
                "fix": fix}
    if kept[0]["cell"] == "NOISE":
        return {"headline": "NOISY_CONTEXT",
                "detail": f"the answer got in at rank {hits[0]['rank']}, but rank 1 is a passage that "
                          "looks right and isn't. The model reads that one first."}
    if kept[0]["cell"] == "HIT":
        return {"headline": "HEALTHY", "detail": "the top-ranked passage answers the question"}
    return {"headline": "HEALTHY", "detail": f"the answer got in at rank {hits[0]['rank']}; rank 1 was not judged"}


def diagnose(rows: list[dict], pool: int, top_k: int, min_sim: float) -> dict:
    """The headline, plus a warning when the judge skipped passages, since a skipped one may be the answer."""
    result = headline(rows, pool, top_k, min_sim)
    skipped = sum(r["cell"] == "UNJUDGED" for r in rows)
    if skipped and result["headline"] != "JUDGE_UNAVAILABLE":
        result["detail"] += f" Careful: the judge skipped {skipped} of {len(rows)} passages."
    return result


def judge_agreement(rows: list[dict], answer_pids: list[str]) -> dict:
    """How often the judge's 'answers' matches the hand labels. A judge needs checking too."""
    judged = [r for r in rows if r["cell"] != "UNJUDGED"]
    wrong = [r["pid"] for r in judged if (r["label"] == "answers") != (r["pid"] in answer_pids)]
    return {"agree": len(judged) - len(wrong), "total": len(judged), "disagreed_on": wrong}


if __name__ == "__main__":
    from src.judge import Verdict

    def case(sims: list[float], labels: list[str]) -> list[dict]:
        """Build rows for a fake query: similarities in rank order, one judge label (or None) per passage."""
        cands = [Candidate(f"p#{i}", "p", "text", s, i + 1) for i, s in enumerate(sims)]
        verdicts = {c.pid: Verdict(label, "") for c, label in zip(cands, labels) if label}
        return build_rows(cands, JudgeResult(verdicts=verdicts), top_k=3, min_sim=0.30)

    def verdict(rows: list[dict]) -> dict:
        """Diagnose fake rows under the same policy they were built with."""
        return diagnose(rows, 10, top_k=3, min_sim=0.30)

    healthy = case([0.8, 0.5, 0.4, 0.2], ["answers", "unrelated", "partial", "unrelated"])
    noisy = case([0.6, 0.5, 0.4, 0.2], ["partial", "answers", "unrelated", "unrelated"])
    miss = case([0.6, 0.5, 0.4, 0.35, 0.33], ["unrelated", "partial", "unrelated", "unrelated", "answers"])
    sim_miss = case([0.6, 0.297, 0.2], ["unrelated", "answers", "unrelated"])
    gap = case([0.5, 0.4, 0.33, 0.2], ["partial", "unrelated", "unrelated", "unrelated"])
    skipped = case([0.5, 0.4, 0.33, 0.2], ["partial", "unrelated", None, "unrelated"])
    lost = case([0.25, 0.19, 0.17], ["unrelated", "unrelated", "unrelated"])

    assert verdict(healthy)["headline"] == "HEALTHY"
    assert verdict(noisy)["headline"] == "NOISY_CONTEXT"
    assert verdict(miss)["fix"] == {"top_k": 5, "min_sim": 0.30, "change": "top-k=5", "extra": 1}
    assert verdict(sim_miss)["fix"]["change"] == "min-sim=0.29", verdict(sim_miss)
    assert sim_miss[1]["retriever_why"] == "sim 0.297 < 0.30"
    assert verdict(gap)["headline"] == "CORPUS_GAP" and "skipped" not in verdict(gap)["detail"]
    assert "skipped 1 of 4" in verdict(skipped)["detail"]
    assert "may not understand the question" in verdict(lost)["detail"]
    assert metrics(noisy) == {"precision": 0.33, "recall": 1.0, "hit": 1, "noise": 2, "miss": 0}
    assert judge_agreement(miss, ["p#4"]) == {"agree": 5, "total": 5, "disagreed_on": []}
    print("OK - all diagnosis rules behave as documented")
