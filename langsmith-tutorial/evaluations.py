"""
evaluations.py
--------------
Creates an evaluation dataset in LangSmith and runs automated
evaluations using the openevals package - the current recommended
approach for LLM-as-judge scoring.

The LangChainStringEvaluator pattern is legacy. This guide uses
the modern openevals.llm.create_llm_as_judge approach documented
in the official LangSmith evaluation quickstart (May 2026).

Run with:
  PYTHONPATH=. uv run python evaluations.py
"""

from dotenv import load_dotenv

from langsmith import Client
from langsmith.evaluation import evaluate
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT

from langchain.agents import create_agent
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

# Initialise the LangSmith client.
# Uses LANGSMITH_API_KEY from the environment automatically.
client = Client()


def create_eval_dataset():
    """
    Create an evaluation dataset in LangSmith with test question-answer pairs.

    Each example has:
      - inputs: the question sent to the agent
      - outputs: the reference answer the evaluator compares against

    Returns:
        The name of the created (or existing) dataset.
    """

    dataset_name = "agent-eval-dataset"

    # Check if the dataset already exists to avoid creating duplicates
    datasets = list(client.list_datasets(dataset_name=dataset_name))
    if datasets:
        print("✅ Dataset already exists - using existing dataset")
        return dataset_name

    # Define the test cases
    examples = [
        {
            "inputs": {"question": "What is LangSmith used for in AI development?"},
            "outputs": {
                "answer": (
                    "LangSmith is used for tracing, monitoring, evaluating, and debugging "
                    "LLM applications. It provides observability into every LLM call, "
                    "tool invocation, and agent step."
                )
            },
        },
        {
            "inputs": {"question": "What is the difference between a trace and a span in LangSmith?"},
            "outputs": {
                "answer": (
                    "A trace is the complete record of one user interaction from start to finish. "
                    "A span is the record of one individual operation within that trace, such as "
                    "one LLM call. A trace contains many spans nested in a tree structure."
                )
            },
        },
        {
            "inputs": {"question": "What is the create_agent function in LangChain used for?"},
            "outputs": {
                "answer": (
                    "create_agent is LangChain's current agent factory function. It builds a "
                    "ReAct agent that loops between an LLM and tools until it produces a final "
                    "answer. It replaces the deprecated langgraph.prebuilt.create_react_agent."
                )
            },
        },
    ]

    # Create the dataset and populate it with examples
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Test questions for the LangSmith tutorial agent",
    )

    # create_examples accepts a list of dicts - more efficient than
    # calling create_example in a loop for large datasets
    client.create_examples(
        inputs=[e["inputs"] for e in examples],
        outputs=[e["outputs"] for e in examples],
        dataset_id=dataset.id,
    )

    print(f"✅ Created dataset '{dataset_name}' with {len(examples)} examples")
    return dataset_name


def agent_target(inputs: dict) -> dict:
    """
    The target function that LangSmith evaluates.

    LangSmith's evaluate() calls this function for each example in the
    dataset, passing the example's inputs dict. The function must return
    a dict whose keys match what the evaluator expects.

    Args:
        inputs: {"question": "..."}  - from the dataset example

    Returns:
        {"answer": "..."}  - the agent's response
    """

    # Build the agent fresh for each evaluation run to avoid state bleed
    search_tool = TavilySearchResults(max_results=2)
    agent = create_agent(
        model="groq:llama-3.3-70b-versatile",
        tools=[search_tool],
        system_prompt="You are a helpful AI assistant. Answer clearly and accurately.",
    )

    question = inputs["question"]

    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )

    return {"answer": result["messages"][-1].content}


def run_evaluation():
    """
    Run the full evaluation suite:
    1. Create or fetch the dataset
    2. Run agent_target on every example
    3. Score each answer using openevals CORRECTNESS_PROMPT
    4. Store results in LangSmith as a named experiment

    The CORRECTNESS_PROMPT evaluator compares the agent's answer
    to the reference answer and returns a correctness score.
    """

    print("\n📋 Setting up evaluation dataset...\n")
    dataset_name = create_eval_dataset()

    print("🔍 Running agent on all dataset examples...\n")

    # create_llm_as_judge builds an LLM-as-judge evaluator.
    # CORRECTNESS_PROMPT is a prebuilt prompt that checks whether
    # the agent's answer is factually correct relative to the reference.
    # model accepts the same "provider:model" format as create_agent.
    correctness_evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        feedback_key="correctness",
        model="groq:llama-3.3-70b-versatile",
    )

    # evaluate() runs agent_target on every example in the dataset,
    # then runs the evaluator on every (prediction, reference) pair,
    # and stores all results in LangSmith under a named experiment.
    evaluate(  # type: ignore[call-overload]
        agent_target,
        data=dataset_name,
        evaluators=[correctness_evaluator],
        experiment_prefix="tutorial-eval",
        num_repetitions=1,
    )

    print("\n✅ Evaluation complete!")
    print(f"📊 View results: LangSmith → Datasets → {dataset_name} → Experiments")


if __name__ == "__main__":
    run_evaluation()