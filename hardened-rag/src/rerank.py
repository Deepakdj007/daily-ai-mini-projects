"""A cross-encoder second pass over the dense candidates.

Inputs:  a question and its dense candidate list
Outputs: the same hits, reordered by a model that reads both texts together

A bi-encoder embeds the question and the passage separately and compares two
vectors, so it can only ever measure topical closeness. A cross-encoder reads
the pair in one forward pass and scores whether the passage actually answers
the question, which is why it fixes sibling confusion.

The registered prediction for this rung is that it helps on clean retrieval and
does nothing for poison, because a poisoned passage is engineered to be maximally
relevant - it opens with the question and then answers it. Reranking rewards
exactly that. If the number comes back worse than the rung below, that is the
finding, not a bug.
"""

from __future__ import annotations

import threading

from src import config
from src.index import Hit

_model = None
_lock = threading.Lock()


def _get_model():
    """Load the cross-encoder once per process."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import CrossEncoder

                _model = CrossEncoder(config.RERANK_MODEL)
    return _model


def rerank(question: str, hits: list[Hit], *, top_k: int = 0) -> list[Hit]:
    """Re-score the candidates as (question, passage) pairs and keep the best."""
    top_k = top_k or config.TOP_K
    if not hits:
        return []
    pairs = [(question, hit.passage.body) for hit in hits]
    scores = _get_model().predict(pairs)
    ordered = sorted(zip(hits, scores), key=lambda pair: float(pair[1]), reverse=True)
    return [hit for hit, _ in ordered[:top_k]]


def warm_up() -> None:
    """Pay the load cost before anything is being timed."""
    _get_model()


if __name__ == "__main__":
    from rich.console import Console

    from src.adversary import build as build_attacks
    from src.corpus import load
    from src.index import candidates, connect

    console = Console()
    corpus = load()
    attacks = {p.pid: p for p in build_attacks(corpus)}
    conn = connect()

    # Two questions answered by the same document: the sibling case a
    # bi-encoder is worst at, because both passages are about wallet caps.
    for question in corpus.questions[:2]:
        dense = candidates(conn, corpus, attacks, question, cond="clean")
        dense_top = dense[:config.TOP_K]
        reranked = rerank(question.question, dense)
        in_dense = any(hit.passage.pid == question.gold_pid for hit in dense_top)
        in_rerank = any(hit.passage.pid == question.gold_pid for hit in reranked)
        console.print(f"[bold]{question.qid}[/] {question.question}")
        console.print(f"  gold in dense top-{config.TOP_K}: {'yes' if in_dense else 'NO'}"
                      f"   after rerank: {'yes' if in_rerank else 'NO'}")

    poisoned = corpus.questions[0]
    dense = candidates(conn, corpus, attacks, poisoned, cond="poison-p", dose=3)
    reranked = rerank(poisoned.question, dense)
    attacks_kept = sum(hit.passage.adversarial for hit in reranked)
    console.print(f"\npoison-p N=3 after rerank: {attacks_kept}/{len(reranked)} "
                  f"passages are the attack")
    conn.close()
