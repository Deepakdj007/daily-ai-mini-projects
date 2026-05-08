# query.py
import io
import torch
from PIL import Image
from pdf2image import convert_from_path
from rag import load_model, load_index
from ingest import pdf_to_images
# query.py (continued)
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from colpali_engine.models import ColQwen2, ColQwen2Processor

load_dotenv()


def retrieve_pages(
    query: str,
    model: ColQwen2,
    processor: ColQwen2Processor,
    embeddings: list[torch.Tensor],
    metadata: list[dict],
    top_k: int = 3,
) -> list[dict]:
    # Encode the query text
    inputs = processor.process_queries([query]).to(model.device)
    
    with torch.no_grad():
        query_emb = model(**inputs)  # shape: (1, n_tokens, embed_dim)
    
    # Score against all pages using MaxSim (late interaction)
    scores = processor.score_multi_vector(query_emb, embeddings)
    
    # scores is shape (1, n_pages) — squeeze the batch dimension
    scores = scores[0].tolist()
    
    # Rank pages by score
    ranked = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True
    )
    
    results = []
    for idx, score in ranked[:top_k]:
        results.append({
            **metadata[idx],
            "score": round(score, 4),
        })
    
    return results

def pil_to_part(img: Image.Image) -> types.Part:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")

def answer_with_gemini(
    query: str,
    page_image: Image.Image,
    page_info: dict,
) -> str:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    
    contents = [
        pil_to_part(page_image),
        f"Based on this document page, answer the following question:\n\n{query}",
    ]
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
    )
    
    return response.text

def run_query(query: str, pdf_path: str, index_dir: str = "index") -> None:
    print(f"\nQuery: {query}\n")
    
    # Load model and index
    model, processor = load_model()
    embeddings, metadata = load_index(index_dir)
    
    # Retrieve top pages
    print("Retrieving relevant pages...")
    results = retrieve_pages(query, model, processor, embeddings, metadata)
    
    print(f"Top match: Page {results[0]['page_number']} (score: {results[0]['score']})")
    
    # Load the actual page image for Gemini
    pages = pdf_to_images(pdf_path)
    top_page_img = pages[results[0]["page_number"] - 1]
    
    # Generate answer
    print("Generating answer with Gemini...")
    answer = answer_with_gemini(query, top_page_img, results[0])
    
    print(f"\n{'='*60}")
    print(f"ANSWER (from page {results[0]['page_number']}):")
    print(f"{'='*60}")
    print(answer)
    print(f"{'='*60}\n")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python query.py 'your question' path/to/document.pdf")
        sys.exit(1)
    
    run_query(query=sys.argv[1], pdf_path=sys.argv[2])  