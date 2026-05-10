# src/server.py
import uuid
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .query_engine import QueryEngine

app = FastAPI(
    title="Annual Report AI",
    description="Multimodal AI for understanding financial documents",
    version="1.0.0",
)

_engine = QueryEngine()  # single instance shared across all requests

class QueryRequest(BaseModel):
    session_id: str
    question: str
    page_numbers: Optional[list[int]] = None

@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    pdf_bytes = await file.read()

    if len(pdf_bytes) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="PDF too large. Max 50MB.")

    session_id = str(uuid.uuid4())

    try:
        result = _engine.ingest(session_id=session_id, pdf_bytes=pdf_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(content=result)

@app.post("/query")
async def query_document(request: QueryRequest):
    try:
        result = _engine.query(
            session_id=request.session_id,
            question=request.question,
            page_numbers=request.page_numbers,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(content=result)


@app.get("/sessions")
async def list_sessions():
    return JSONResponse(content={"sessions": _engine.list_sessions()})


@app.get("/health")
async def health():
    return {"status": "ok"}

