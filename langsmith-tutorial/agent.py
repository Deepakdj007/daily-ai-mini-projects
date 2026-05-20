"""
agent.py
--------
A LangChain ReAct agent with full LangSmith observability.
The agent can search the web using Tavily to answer questions
that require up-to-date information.

Stack:
  - langchain.agents.create_agent (current API, replaces deprecated
    langgraph.prebuilt.create_react_agent)
  - Groq llama-3.3-70b-versatile (free LLM)
  - Tavily (web search tool, free tier)
  - LangSmith (automatic tracing - no extra code)

Run:
  PYTHONPATH=. uv run python agent.py
"""

import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage

# Load .env - sets LANGSMITH_TRACING, LANGSMITH_API_KEY, LANGSMITH_PROJECT
load_dotenv()


def build_agent():
    """
    Build and return a LangChain ReAct agent using the current create_agent API.

    create_agent accepts:
      - model: a model string in the format "provider:model-name"
                or just "model-name" if the provider is inferred from env vars.
                For Groq, use "groq:llama-3.3-70b-versatile".
      - tools: a list of tool functions or LangChain tool objects.
      - system_prompt: a string system prompt (replaces the separate prompt object).

    The agent runs the ReAct loop:
      Reason → Act (call tool) → Observe → Reason again → Final answer

    Returns:
        A compiled LangGraph StateGraph (the agent runtime).
    """

    # Tavily search tool - max_results=3 keeps token usage reasonable
    search_tool = TavilySearchResults(max_results=3)

    # create_agent is the current, non-deprecated agent factory.
    # Note: the `model` parameter takes a model string, not a ChatGroq instance.
    # For Groq models, use the "groq:" prefix so LangChain knows which provider to use.
    agent = create_agent(
        model="groq:llama-3.3-70b-versatile",
        tools=[search_tool],
        system_prompt=(
            "You are a helpful AI research assistant. "
            "Search for recent, reliable information and cite specific facts."
        ),
    )

    return agent


def run_agent(question: str) -> str:
    """
    Run the agent on a given question and return the final answer.

    Because LANGSMITH_TRACING=true is set, LangChain automatically sends
    the full trace to LangSmith. Every node - the initial LLM reasoning,
    each Tavily search call, and the final LLM response - appears as a
    separate named span.

    Args:
        question: The user's question.

    Returns:
        The agent's final answer as a string.
    """

    agent = build_agent()

    # invoke() accepts a messages list - same interface as before
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )

    # The final message in the messages list is the agent's answer
    return result["messages"][-1].content


def main():
    """
    Ask the agent a question that requires real-time web search.
    After running, open LangSmith to see the full trace.
    """

    question = "What are the top 3 AI engineering skills companies are hiring for in 2025? Give me recent data."

    print(f"\n🤖 Agent Question: {question}")
    print("\n⏳ Running agent...\n")
    print("-" * 60)

    answer = run_agent(question)

    print("\n✅ Final Answer:")
    print(answer)
    print("-" * 60)
    print("\n📊 View the full trace at: https://smith.langchain.com")
    print(f"   Project: {os.getenv('LANGSMITH_PROJECT', 'default')}")


if __name__ == "__main__":
    main()