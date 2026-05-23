# evaluate.py
# Loads RAG outputs and scores them using RAGAS v0.4 collections metrics.

# load_dotenv must be the very first thing — before any other imports.
from dotenv import load_dotenv
load_dotenv()

import os
import json
import asyncio
import instructor
from openai import AsyncOpenAI

from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.llms.base import InstructorLLM, InstructorModelArgs
from ragas.embeddings import GoogleEmbeddings
from ragas.metrics.collections import (
    Faithfulness,     # hallucination detector
    AnswerRelevancy,  # does the answer address the question?
    ContextPrecision, # is the retrieved context focused?
    ContextRecall,    # did retrieval find everything needed?
)


def load_rag_outputs(filepath: str) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def build_evaluation_dataset(rag_outputs: list[dict]) -> EvaluationDataset:
    """
    Converts RAG outputs into a RAGAS EvaluationDataset.
    question → user_input, answer → response, contexts → retrieved_contexts.
    """
    samples = [
        SingleTurnSample(
            user_input=row["question"],
            response=row["answer"],
            retrieved_contexts=row["contexts"],
            reference=row["reference"],
        )
        for row in rag_outputs
    ]
    return EvaluationDataset(samples=samples)


def setup_evaluator():
    """
    Sets up the RAGAS evaluator LLM and embeddings.

    llm_factory defaults to Mode.TOOLS (function calling). Gemini's
    OpenAI-compatible endpoint accepts tool calls but returns empty
    statements lists, causing faithfulness = NaN. Mode.JSON avoids tool
    calling entirely — the model generates raw JSON that instructor parses.
    """
    raw_client = AsyncOpenAI(
        api_key=os.environ["GOOGLE_API_KEY"],
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    patched_client = instructor.from_openai(raw_client, mode=instructor.Mode.MD_JSON)
    llm = InstructorLLM(
        client=patched_client,
        model="gemini-2.5-flash",
        provider="openai",
        model_args=InstructorModelArgs(max_tokens=4096),
    )
    embeddings = GoogleEmbeddings(model="gemini-embedding-001")
    return llm, embeddings


async def _score_all(samples: list[SingleTurnSample], llm, embeddings) -> dict[str, float]:
    """
    Calls ascore() on each sample for all four metrics and averages the results.

    ragas.metrics.collections metrics cannot be used with ragas.evaluate() —
    they inherit from SimpleBaseMetric, not Metric, and evaluate() rejects them
    at runtime. We call ascore() directly instead.

    ascore() argument names per metric:
      Faithfulness     : user_input, response, retrieved_contexts
      AnswerRelevancy  : user_input, response
      ContextPrecision : user_input, reference, retrieved_contexts
      ContextRecall    : user_input, retrieved_contexts, reference
    """
    faithfulness = Faithfulness(llm=llm)
    answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
    context_precision = ContextPrecision(llm=llm)
    context_recall = ContextRecall(llm=llm)

    f_scores, ar_scores, cp_scores, cr_scores = [], [], [], []

    for i, sample in enumerate(samples):
        print(f"  Scoring sample {i + 1}/{len(samples)}...")

        # All four fields are required — these asserts will catch any samples
        # that were built without a reference (needed by precision and recall)
        assert sample.user_input is not None
        assert sample.response is not None
        assert sample.retrieved_contexts is not None
        assert sample.reference is not None

        # Score all four metrics concurrently for this sample
        f, ar, cp, cr = await asyncio.gather(
            faithfulness.ascore(
                user_input=sample.user_input,
                response=sample.response,
                retrieved_contexts=sample.retrieved_contexts,
            ),
            answer_relevancy.ascore(
                user_input=sample.user_input,
                response=sample.response,
            ),
            context_precision.ascore(
                user_input=sample.user_input,
                reference=sample.reference,
                retrieved_contexts=sample.retrieved_contexts,
            ),
            context_recall.ascore(
                user_input=sample.user_input,
                retrieved_contexts=sample.retrieved_contexts,
                reference=sample.reference,
            ),
        )
        f_scores.append(f.value)
        ar_scores.append(ar.value)
        cp_scores.append(cp.value)
        cr_scores.append(cr.value)

    def avg(lst: list[float]) -> float:
        import math
        valid = [x for x in lst if not math.isnan(x)]
        return round(sum(valid) / len(valid), 4) if valid else float("nan")

    return {
        "faithfulness": avg(f_scores),
        "answer_relevancy": avg(ar_scores),
        "context_precision": avg(cp_scores),
        "context_recall": avg(cr_scores),
    }


def run_evaluation(dataset: EvaluationDataset, llm, embeddings) -> dict[str, float]:
    print("Running RAGAS evaluation (1-3 minutes)...")
    # Filter to SingleTurnSample only — EvaluationDataset can hold mixed types
    samples = [s for s in dataset.samples if isinstance(s, SingleTurnSample)]
    return asyncio.run(_score_all(samples, llm, embeddings))


if __name__ == "__main__":
    print("Loading RAG outputs...")
    rag_outputs = load_rag_outputs("rag_outputs.json")
    print(f"Loaded {len(rag_outputs)} samples.\n")

    print("Building evaluation dataset...")
    dataset = build_evaluation_dataset(rag_outputs)
    print("Dataset ready.\n")

    print("Setting up evaluator (Gemini 2.5 Flash)...")
    llm, embeddings = setup_evaluator()
    print("Evaluator ready.\n")

    scores = run_evaluation(dataset, llm, embeddings)

    print("\n" + "=" * 60)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 60)
    for name, score in scores.items():
        print(f"  {name}: {score:.4f}")

    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)
    print("\nResults saved to eval_results.json")