from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os

from extract import extract_from_folder
from graph_db import get_driver, load_graph, get_full_graph, clear_graph
from rag import answer_question

load_dotenv()

app = FastAPI(title="GraphRAG API", version="1.0.0")

# CORS middleware is required because the React frontend (localhost:5173)
# and FastAPI backend (localhost:8000) are on different ports.
# Without this, the browser blocks all requests from the frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite's default dev server port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for the query endpoint — FastAPI uses this for 
# automatic request body validation and API documentation
class QueryRequest(BaseModel):
    question: str

class IngestRequest(BaseModel):
    folder_path: str = "documents"  # Default to "documents" folder
    clear_first: bool = True         # Wipe existing graph before ingest

@app.get("/")
def root():
    """Health check endpoint"""
    return {"status": "GraphRAG API is running"}

@app.post("/api/ingest")
def ingest_documents(request: IngestRequest):
    """
    Read all .txt files from the specified folder, extract entities and 
    relationships using Groq, and write the knowledge graph to Neo4j.
    
    This is the pipeline trigger — call this once before querying.
    Expect it to take 30-60 seconds depending on document count.
    """
    try:
        # Extract graph data from documents
        extracted = extract_from_folder(request.folder_path)
        
        if not extracted["entities"]:
            raise HTTPException(
                status_code=422,
                detail=f"No entities extracted. Check that {request.folder_path} contains .txt files."
            )
        
        # Get a Neo4j driver (verifies connection)
        driver = get_driver()
        
        # Optionally clear the existing graph first
        if request.clear_first:
            clear_graph(driver)
        
        driver.close()
        
        # Load the extracted graph into Neo4j
        load_graph(extracted)
        
        return {
            "status": "success",
            "entities_count": len(extracted["entities"]),
            "relationships_count": len(extracted["relationships"])
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/api/graph")
def get_graph():
    """
    Return the full knowledge graph in a format ready for react-force-graph-2d.
    {"nodes": [{"id": "...", "name": "...", "type": "..."}], 
     "links": [{"source": "...", "target": "...", "relation": "..."}]}
    """
    try:
        driver = get_driver()
        graph = get_full_graph(driver)
        driver.close()
        return graph
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/api/query")
def query_graph(request: QueryRequest):
    """
    Ask a question. The engine extracts entities from the question,
    retrieves the relevant subgraph, and returns a grounded answer.
    
    Response includes:
    - answer: the LLM's answer grounded in the graph
    - entities: which entities were found in the question
    - subgraph: the graph context used to generate the answer
    """
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty")
    
    try:
        result = answer_question(request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.delete("/api/graph")
def delete_graph():
    """Clear all nodes and relationships from the graph database."""
    try:
        driver = get_driver()
        clear_graph(driver)
        driver.close()
        return {"status": "Graph cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
