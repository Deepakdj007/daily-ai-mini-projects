import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
import chromadb
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

load_dotenv()

class AgentState(TypedDict):
    question: str           # the user's original question
    rewritten_question: str # a rephrased version for retry
    documents: list[str]    # chunks retrieved from ChromaDB
    answer: str             # the generated answer
    grade: str              # "pass" or "fail"
    retry_count: int        # how many retries have happened

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.environ.get("GROQ_API_KEY")
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    metadata={"hnsw:space": "cosine"}
)

def retrieve(state: AgentState) -> AgentState:
    """Search ChromaDB for documents relevant to the current question."""
    question = state.get("rewritten_question") or state["question"]

    print(f"\n[RETRIEVE] Searching for: {question}")

    try:
        results = collection.query(
            query_texts=[question],
            n_results=3
        )
        documents = results["documents"][0]
        print(f"[RETRIEVE] Found {len(documents)} chunks.")
    except Exception as error:
        print(f"[RETRIEVE] ChromaDB query failed: {error}")
        documents = []

    return {**state, "documents": documents}

def generate(state: AgentState) -> AgentState:
    """Generate an answer using retrieved documents as context."""
    question = state.get("rewritten_question") or state["question"]
    documents = state.get("documents", [])

    print(f"\n[GENERATE] Generating answer for: {question}")

    if not documents:
        return {**state, "answer": "No relevant documents were found."}

    context = "\n\n".join([f"Document {i+1}:\n{doc}"
                           for i, doc in enumerate(documents)])

    system_prompt = """You are a helpful assistant that answers questions
based strictly on the provided documents. If the documents do not contain
enough information to answer the question, say so honestly.
Do not use any knowledge outside of the provided documents."""

    user_prompt = f"""Documents:
{context}

Question: {question}

Answer based only on the documents above:"""

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        answer = response.content
        print(f"[GENERATE] Answer generated ({len(answer)} chars).")
    except Exception as error:
        print(f"[GENERATE] LLM call failed: {error}")
        answer = "An error occurred while generating the answer."

    return {**state, "answer": answer}

def grade_answer(state: AgentState) -> AgentState:
    """Grade whether the answer is grounded in the retrieved documents."""
    question = state["question"]
    documents = state.get("documents", [])
    answer = state.get("answer", "")

    print(f"\n[GRADE] Evaluating answer quality...")

    context = "\n\n".join(documents) if documents else "No documents."

    grading_prompt = f"""You are a strict grader evaluating whether an answer
is grounded in the provided documents.

Documents:
{context}

Question: {question}

Answer: {answer}

Evaluate this answer on two criteria:
1. Is the answer directly supported by information in the documents?
2. Does the answer avoid making claims not found in the documents?

Respond with ONLY one word: PASS or FAIL.
PASS means the answer is well-supported by the documents.
FAIL means the answer contains unsupported claims or is not relevant."""

    try:
        response = llm.invoke([HumanMessage(content=grading_prompt)])
        grade_raw = response.content.strip().upper()
        grade = "pass" if "PASS" in grade_raw else "fail"
        print(f"[GRADE] Result: {grade.upper()}")
    except Exception as error:
        print(f"[GRADE] Grading call failed: {error}")
        grade = "fail"

    return {**state, "grade": grade}

def rewrite_question(state: AgentState) -> AgentState:
    """Rewrite the question to improve retrieval on the next attempt."""
    original_question = state["question"]
    retry_count = state.get("retry_count", 0)

    print(f"\n[REWRITE] Rewriting question (attempt {retry_count + 1})...")

    rewrite_prompt = f"""The following question did not retrieve useful documents
from a knowledge base. Rewrite the question to be more specific and use
different vocabulary that might match relevant documents better.

Original question: {original_question}

Write only the rewritten question, nothing else:"""

    try:
        response = llm.invoke([HumanMessage(content=rewrite_prompt)])
        rewritten = response.content.strip()
        print(f"[REWRITE] New question: {rewritten}")
    except Exception as error:
        print(f"[REWRITE] Rewrite failed: {error}")
        rewritten = original_question

    return {
        **state,
        "rewritten_question": rewritten,
        "retry_count": retry_count + 1
    }

def give_up(state: AgentState) -> AgentState:
    """Return an honest 'I don't know' after all retries are exhausted."""
    print(f"\n[GIVE UP] Could not find a reliable answer after retries.")

    honest_answer = (
        "I was unable to find a reliable answer to your question "
        "in the available documents. The knowledge base may not contain "
        "information about this topic. Please try rephrasing your question "
        "or consult additional sources."
    )

    return {**state, "answer": honest_answer}

def route_after_grading(state: AgentState) -> str:
    """Decide where to go after grading the answer."""
    grade = state.get("grade", "fail")
    retry_count = state.get("retry_count", 0)
    max_retries = 2

    if grade == "pass":
        return "end"
    elif retry_count >= max_retries:
        return "give_up"
    else:
        return "rewrite"
    
def build_graph() -> StateGraph:
    """Assemble all nodes and edges into the LangGraph."""
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("grade_answer", grade_answer)
    graph.add_node("rewrite_question", rewrite_question)
    graph.add_node("give_up", give_up)

    # Set the entry point - first node to run
    graph.set_entry_point("retrieve")

    # Add fixed edges (always go from A to B)
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "grade_answer")
    graph.add_edge("rewrite_question", "retrieve")  # the retry loop!
    graph.add_edge("give_up", END)

    # Add the conditional edge after grading
    graph.add_conditional_edges(
        "grade_answer",
        route_after_grading,
        {
            "end": END,
            "give_up": "give_up",
            "rewrite": "rewrite_question"
        }
    )

    return graph.compile()

if __name__ == "__main__":
    print("Self-Healing RAG System")
    print("=" * 40)
    print("Type your question and press Enter.")
    print("Type 'quit' to exit.\n")

    rag_graph = build_graph()

    while True:
        user_question = input("Your question: ").strip()

        if user_question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not user_question:
            continue

        initial_state: AgentState = {
            "question": user_question,
            "rewritten_question": "",
            "documents": [],
            "answer": "",
            "grade": "",
            "retry_count": 0
        }

        try:
            final_state = rag_graph.invoke(initial_state)
            print(f"\n{'=' * 40}")
            print(f"ANSWER:\n{final_state['answer']}")
            print(f"{'=' * 40}\n")
        except Exception as error:
            print(f"An error occurred: {error}")