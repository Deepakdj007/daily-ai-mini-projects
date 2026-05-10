# src/gemini_client.py
import os
from dataclasses import dataclass

import google.genai as genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

@dataclass
class PageAnalysis:
    page_number: int
    raw_text: str        # everything the model extracted from this page
    has_charts: bool
    has_tables: bool
    key_numbers: list[str]  # lines containing financial figures

_ANALYST_SYSTEM_PROMPT = """
You are a financial document analyst specializing in annual reports.
When shown a page image, you must:
1. Extract ALL visible text, including text inside charts, axes, and tables
2. Identify every numerical value and what it represents
3. Describe chart types and what trends they show
4. Read every table cell accurately
5. Note any year-over-year comparisons visible

Be precise with numbers. Never skip a data point because it is small or in a legend.
Output everything you see — completeness is more important than brevity.
""".strip()

class GeminiVisionClient:
    MODEL = "gemini-2.5-flash"

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Set it in your .env file."
            )
        self._client = genai.Client(api_key=api_key)
    
    def analyze_page(self, page: "PageImage") -> PageAnalysis:
        """Send one page image to Gemini and return structured analysis."""
        image_part = types.Part.from_bytes(
            data=page.image_bytes,
            mime_type=page.mime_type,
        )
        text_part = types.Part.from_text(
            text=(
                f"Analyze page {page.page_number} of this annual report. "
                "Extract all text, numbers, chart data, and table contents."
            )
        )

        response = self._client.models.generate_content(
            model=self.MODEL,
            contents=[image_part, text_part],
            config=types.GenerateContentConfig(
                system_instruction=_ANALYST_SYSTEM_PROMPT,
                temperature=0.1,
                max_output_tokens=2048,
            ),
        )
        return self._parse_analysis(page.page_number, response.text)
    
    def _parse_analysis(self, page_num: int, text: str) -> PageAnalysis:
        text_lower = text.lower()

        has_charts = any(w in text_lower for w in [
            "chart", "graph", "bar", "line", "pie", "trend", "axis", "legend"
        ])
        has_tables = any(w in text_lower for w in [
            "table", "row", "column", "header", "cell"
        ])

        key_numbers = [
            line.strip()
            for line in text.split("\n")
            if any(s in line for s in ["$", "%", "₹", "€", "£", "million", "billion", "crore"])
            and len(line.strip()) > 5
        ]

        return PageAnalysis(
            page_number=page_num,
            raw_text=text,
            has_charts=has_charts,
            has_tables=has_tables,
            key_numbers=key_numbers[:20],
        )
    
    def answer_question(
        self,
        question: str,
        pages: list["PageImage"],
        page_context: list[PageAnalysis],
    ) -> str:
        parts: list[types.Part] = []

        # 1. Text context from prior extraction — gives the model a head start
        context_summary = "\n\n".join(
            f"[Page {a.page_number}]\n{a.raw_text}"
            for a in page_context
        )
        parts.append(types.Part.from_text(
            text=f"Here is what was extracted from the document:\n\n{context_summary}"
        ))

        # 2. Actual page images for visual grounding
        for page in pages[:5]:  # cap at 5 — keeps latency low
            parts.append(types.Part.from_bytes(
                data=page.image_bytes,
                mime_type=page.mime_type,
            ))

        # 3. The question — always last
        parts.append(types.Part.from_text(
            text=f"\nBased on all the above, answer this question precisely:\n{question}"
        ))

        response = self._client.models.generate_content(
            model=self.MODEL,
            contents=parts,
            config=types.GenerateContentConfig(
                system_instruction=_ANALYST_SYSTEM_PROMPT,
                temperature=0.1,
                max_output_tokens=1024,
            ),
        )
        return response.text