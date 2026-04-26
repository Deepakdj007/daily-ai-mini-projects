import json
import re
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq()

EXTRACTION_PROMPT = """You are a knowledge graph extraction expert.

Read the text below and extract:
1. ENTITIES — named things (people, organizations, products, concepts, places, events)
2. RELATIONSHIPS — how those entities connect to each other

Return ONLY valid JSON. No explanation, no markdown, no code fences.

Format:
{
  "entities": [
    {"id": "e1", "name": "Entity Name", "type": "Person|Organization|Product|Concept|Place|Event"}
  ],
  "relationships": [
    {"source": "e1", "target": "e2", "relation": "RELATION_TYPE"}
  ]
}

Rules:
- Use short, uppercase relation types like WORKS_AT, CREATED, ACQUIRED, USES, LOCATED_IN
- Every relationship must reference entity ids that exist in the entities list
- ONLY extract relationships that are explicitly stated in a single sentence — do NOT infer or combine facts across sentences
- Do NOT add a relationship just because two entities appear near each other
- Deduplicate: if the same entity appears multiple times, use the same id

TEXT:
"""

def extract_graph_from_text(text: str) -> dict:
    """
    Send a document to Groq and return extracted entities + relationships as a dict.
    Returns {"entities": [...], "relationships": [...]} or empty lists on failure.
    """
    # Truncate very long documents to stay within token limits
    # llama-3.3-70b has a 128K context window but we keep extractions focused
    max_chars = 4000
    if len(text) > max_chars:
        text = text[:max_chars]
        print(f"Document truncated to {max_chars} characters")
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": EXTRACTION_PROMPT + text
            }
        ],
        temperature=0.1,  # Low temperature = more consistent, structured output
        max_tokens=2000   # Enough for a full graph extraction
    )

    # Get the raw text response from the LLM
    raw = response.choices[0].message.content.strip()
    
    # Sometimes models wrap JSON in markdown code fences despite instructions.
    # This regex strips ```json ... ``` or ``` ... ``` wrappers.
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    
    try:
        data = json.loads(raw)
        entities = data.get("entities", [])
        relationships = data.get("relationships", [])
        print(f"  Extracted {len(entities)} entities, {len(relationships)} relationships")
        return {"entities": entities, "relationships": relationships}
    except json.JSONDecodeError as e:
        # If the LLM returns malformed JSON, log it and return empty data
        # so the pipeline can continue with other documents
        print(f"  JSON parse error: {e}")
        print(f"  Raw response: {raw[:200]}")  # Show first 200 chars for debugging
        return {"entities": [], "relationships": []}
    
def extract_from_folder(folder_path: str) -> dict:
    """
    Read all .txt files from a folder and extract a combined graph.
    Returns a merged dict with deduplicated entities and all relationships.
    """
    all_entities = {}   # key: entity name (lowercase) → value: entity dict
    all_relationships = []
    
    # Verify the folder exists before trying to read from it
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return {"entities": [], "relationships": []}
    
    txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    
    if not txt_files:
        print(f"No .txt files found in {folder_path}")
        return {"entities": [], "relationships": []}
    
    print(f"Found {len(txt_files)} documents to process")
    
    for filename in txt_files:
        filepath = os.path.join(folder_path, filename)
        print(f"\nProcessing: {filename}")
        
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        
        result = extract_graph_from_text(text)
        
        # Deduplicate entities by name (case-insensitive)
        # If two documents mention "OpenAI", they become the same node
        for entity in result["entities"]:
            key = entity["name"].lower()
            if key not in all_entities:
                all_entities[key] = entity
        
        all_relationships.extend(result["relationships"])
    
    entities_list = list(all_entities.values())
    print(f"\n📊 Total: {len(entities_list)} unique entities, {len(all_relationships)} relationships")
    
    return {"entities": entities_list, "relationships": all_relationships}