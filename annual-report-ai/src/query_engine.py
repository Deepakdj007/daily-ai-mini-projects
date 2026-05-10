# src/query_engine.py
from dataclasses import dataclass, field

from .pdf_processor import PDFProcessor, PageImage
from .gemini_client import GeminiVisionClient, PageAnalysis

@dataclass
class DocumentSession:
    """A processed document held in memory for repeated querying."""
    pdf_path: str
    pages: list[PageImage] = field(default_factory=list)
    analyses: list[PageAnalysis] = field(default_factory=list)
    total_pages: int = 0

    @property
    def is_ready(self) -> bool:
        return len(self.analyses) > 0
    
class QueryEngine:
    def __init__(self):
        self._processor = PDFProcessor()
        self._vision = GeminiVisionClient()
        self._sessions: dict[str, DocumentSession] = {}
    
    def ingest(self, session_id: str, pdf_bytes: bytes) -> dict:
        """Process a PDF and cache all page analyses."""
        pages = self._processor.load_bytes(pdf_bytes)
        analyses = []

        print(f"[QueryEngine] Ingesting {len(pages)} pages...")
        for page in pages:
            print(f"  → Analyzing page {page.page_number}/{len(pages)}")
            analysis = self._vision.analyze_page(page)
            analyses.append(analysis)

        self._sessions[session_id] = DocumentSession(
            pdf_path=session_id,
            pages=pages,
            analyses=analyses,
            total_pages=len(pages),
        )

        return {
            "session_id": session_id,
            "total_pages": len(pages),
            "pages_with_charts": [a.page_number for a in analyses if a.has_charts],
            "pages_with_tables": [a.page_number for a in analyses if a.has_tables],
            "status": "ready",
        }
    
    def query(
        self,
        session_id: str,
        question: str,
        page_numbers: list[int] | None = None,
    ) -> dict:
        session = self._sessions.get(session_id)
        if not session:
            raise KeyError(f"No session found: {session_id}. Call ingest() first.")
        if not session.is_ready:
            raise RuntimeError("Document is still being processed.")

        if page_numbers:
            analyses = [a for a in session.analyses if a.page_number in page_numbers]
            pages = [p for p in session.pages if p.page_number in page_numbers]
        else:
            analyses = session.analyses
            pages = session.pages

        answer = self._vision.answer_question(
            question=question,
            pages=pages,
            page_context=analyses,
        )

        return {
            "question": question,
            "answer": answer,
            "pages_referenced": [a.page_number for a in analyses],
            "model": GeminiVisionClient.MODEL,
        }

    def list_sessions(self) -> list[dict]:
        return [
            {"session_id": sid, "total_pages": s.total_pages, "ready": s.is_ready}
            for sid, s in self._sessions.items()
        ]