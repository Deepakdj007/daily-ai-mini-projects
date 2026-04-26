from groq import Groq
from graph_db import get_driver, get_subgraph
from dotenv import load_dotenv
import json
import re

load_dotenv()

client = Groq()

def extract_query_entities(question: str) -> list[str]:
    """
    Ask Groq to identify which entities in the question we should
    use as the starting point for graph traversal.
    Returns a list of entity name strings.
    """
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": f"""Extract the key entities (people, organizations, products, places, concepts) 
from this question. Return ONLY a JSON array of strings. No explanation.

Question: {question}

Example output: ["OpenAI", "GPT-4", "Microsoft"]"""
            }
        ],
        temperature=0.1,
        max_tokens=200
    )
    
    raw = response.choices[0].message.content.strip()
    # Strip code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    
    try:
        entities = json.loads(raw)
        # Make sure we got a list of strings
        if isinstance(entities, list):
            return [str(e) for e in entities]
        return []
    except json.JSONDecodeError:
        # Fallback: split by comma if JSON fails
        return [e.strip().strip('"') for e in raw.split(",") if e.strip()]
    
def format_graph_as_context(subgraph: dict) -> str:
    """
    Convert the subgraph dict into a readable text representation
    that the LLM can reason over.
    
    Instead of raw Cypher or JSON, we give the LLM plain English triples.
    This consistently produces better answers than passing raw JSON.
    """
    if not subgraph["nodes"] and not subgraph["links"]:
        return "No relevant information found in the knowledge graph."
    
    lines = []
    
    lines.append("=== Knowledge Graph Context ===\n")
    
    lines.append("Entities found:")
    for node in subgraph["nodes"]:
        lines.append(f"  - {node['name']} (type: {node['type']})")
    
    lines.append("\nRelationships:")
    for link in subgraph["links"]:
        # Format as a human-readable triple: "OpenAI CREATED GPT-4"
        lines.append(f"  - {link['source']} --{link['relation']}--> {link['target']}")
    
    return "\n".join(lines)

def answer_question(question: str) -> dict:
    """
    Full GraphRAG query pipeline:
    1. Extract entities from the question
    2. Retrieve the relevant subgraph from Neo4j
    3. Format the subgraph as context
    4. Ask Groq to answer using only that context
    
    Returns a dict with the answer and the source subgraph for display.
    """
    print(f"\n❓ Question: {question}")
    
    # Step 1: Identify entities in the question
    entities = extract_query_entities(question)
    print(f"🔍 Query entities: {entities}")
    
    if not entities:
        return {
            "answer": "I could not identify any entities in your question. Please try rephrasing.",
            "entities": [],
            "subgraph": {"nodes": [], "links": []}
        }
    
    # Step 2: Retrieve the subgraph around those entities
    driver = get_driver()
    subgraph = get_subgraph(driver, entities, hops=2)
    driver.close()
    
    print(f"📊 Subgraph: {len(subgraph['nodes'])} nodes, {len(subgraph['links'])} relationships")
    
    # Step 3: Format the subgraph as context text
    context = format_graph_as_context(subgraph)
    
    # Step 4: Ask Groq to answer using only the graph context
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """You are a knowledge graph assistant. You answer questions strictly based 
on the graph context provided. If the context does not contain enough information to answer, 
say so clearly. Do not make up facts. Cite which relationships support your answer."""
            },
            {
                "role": "user",
                "content": f"{context}\n\nQuestion: {question}\n\nAnswer based only on the graph context above:"
            }
        ],
        temperature=0.3,  # Slightly higher than extraction — we want articulate answers
        max_tokens=500
    )
    
    answer = response.choices[0].message.content.strip()
    print(f"💬 Answer: {answer[:100]}...")
    
    return {
        "answer": answer,
        "entities": entities,
        "subgraph": subgraph
    }

