"""
Demonstrates the @traceable decorator for custom trace names.
Import and apply it to any function you want to appear as a
named span in the LangSmith dashboard.
"""

from dotenv import load_dotenv
from langsmith import traceable
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()


# The @traceable decorator wraps this function in a named span.
# The 'name' argument controls what appears in the dashboard.
# The 'run_type' argument sets the span type (llm, chain, tool, retriever).
@traceable(name="Answer Question", run_type="chain")
def answer_question(question: str) -> str:
    """
    Wraps the LLM call in a named trace span.
    Appears as 'Answer Question' in the LangSmith dashboard
    instead of the default class name.

    Args:
        question: The question to answer.

    Returns:
        The model's answer as a string.
    """

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=1024,
    )

    messages = [
        SystemMessage(content="You are a helpful AI assistant."),
        HumanMessage(content=question),
    ]

    response = llm.invoke(messages)
    return response.content


if __name__ == "__main__":
    result = answer_question("What is LangSmith used for?")
    print(result)