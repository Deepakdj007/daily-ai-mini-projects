# backend/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv
from rag import build_csv_context
import os
import uuid

load_dotenv()  # reads GEMINI_API_KEY from .env into the environment

app = FastAPI(title="CSV Analyst AI")

# CORS — browsers block cross-origin requests by default.
# This tells the browser that requests from the React dev server (port 5173) are allowed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store: session_id -> csv context string.
# In production, replace this with Redis or a database.
sessions: dict[str, str] = {}

# Create the Gemini client once at startup.
# client.aio is the async sub-client — use this in FastAPI async endpoints
# so we don't block the event loop while waiting for the API.
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

MODEL = "gemini-2.5-flash"  # free tier: 10 RPM, 250 RPD


# ---------------------------------------------------------------------------
# Upload endpoint
# ---------------------------------------------------------------------------


@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """
    Receives a CSV file, builds a RAG context string from it,
    stores it under a new session_id, and returns that session_id.
    The frontend sends it with every subsequent chat message.
    """
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    contents = await file.read()

    try:
        context = build_csv_context(contents)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {exc}")

    session_id = str(uuid.uuid4())
    sessions[session_id] = context

    return {"session_id": session_id, "message": "CSV uploaded and ready."}


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Receives a user message + session_id.
    Retrieves the stored CSV context, injects it into the prompt,
    and streams Gemini's response token-by-token back to the client.
    """
    context = sessions.get(request.session_id)
    if context is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Upload a CSV first.",
        )

    # RAG step: the CSV context is injected into the prompt so Gemini
    # "knows" the data and can answer questions about it.
    full_prompt = f"""You are an expert data analyst. The user has uploaded a CSV file.
Here is the complete data context:

{context}

Answer questions about this data clearly and concisely.
Cite specific numbers, column names, or row values when relevant.
If a question cannot be answered from the provided data, say so explicitly.

User question: {request.message}"""

    async def stream_response():
        stream = await client.aio.models.generate_content_stream(
            model=MODEL,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=1024,
            ),
        )

        async for chunk in stream:
            if chunk.text:
                yield chunk.text

    # StreamingResponse wraps our async generator and sends each yielded chunk
    # immediately using HTTP chunked transfer encoding.
    # The browser reads it token-by-token with the ReadableStream API.
    return StreamingResponse(stream_response(), media_type="text/plain")
