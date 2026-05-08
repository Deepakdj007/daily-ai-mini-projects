# ingest.py
import json
import pathlib
import numpy as np
import torch
from pdf2image import convert_from_path
from PIL import Image
from colpali_engine.models import ColQwen2, ColQwen2Processor
from rag import load_model

POPPLER_PATH = r"C:\poppler\poppler-26.02.0\Library\bin"

def pdf_to_images(pdf_path: str, dpi: int = 150) -> list[Image.Image]:
    pages = convert_from_path(pdf_path, dpi=dpi, fmt="RGB", poppler_path=POPPLER_PATH)
    return pages

def embed_pages(
    pages: list[Image.Image],
    model: ColQwen2,
    processor: ColQwen2Processor,
    batch_size: int = 2,
) -> list[torch.Tensor]:
    all_embeddings = []
    
    for i in range(0, len(pages), batch_size):
        batch = pages[i : i + batch_size]
        
        inputs = processor.process_images(batch).to(model.device)
        
        with torch.no_grad():
            embeddings = model(**inputs)
        
        # Move to CPU and detach before storing
        # Each embedding is shape (n_patches, embed_dim)
        for emb in embeddings:
            all_embeddings.append(emb.cpu().detach())
        
        print(f"Embedded pages {i+1} to {min(i+batch_size, len(pages))}")
    
    return all_embeddings

def build_index(pdf_path: str, index_dir: str = "index") -> None:
    index_path = pathlib.Path(index_dir)
    index_path.mkdir(exist_ok=True)
    
    print(f"Converting {pdf_path} to images...")
    pages = pdf_to_images(pdf_path)
    print(f"Found {len(pages)} pages")
    
    print("Loading model...")
    model, processor = load_model()
    
    print("Embedding pages...")
    embeddings = embed_pages(pages, model, processor)
    
    # Save each page embedding as a separate .npy file
    pdf_name = pathlib.Path(pdf_path).name
    for idx, emb in enumerate(embeddings):
        np.save(index_path / f"emb_{idx}.npy", emb.numpy())
    
    # Save metadata: maps embedding index to source page info
    metadata = [
        {"index": idx, "pdf": pdf_name, "page_number": idx + 1}
        for idx in range(len(embeddings))
    ]
    with open(index_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Index saved: {len(embeddings)} pages in '{index_dir}/'")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ingest.py path/to/document.pdf")
        sys.exit(1)
    
    build_index(pdf_path=sys.argv[1])

