# rag.py
import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor
import json
import pathlib
import numpy as np

def load_model() -> tuple[ColQwen2, ColQwen2Processor]:
    model_name = "vidore/colqwen2-v1.0"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    model = ColQwen2.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
    ).eval()
    
    processor = ColQwen2Processor.from_pretrained(model_name)
    
    print(f"Model loaded on {device} ({dtype})")
    return model, processor

def load_index(index_dir: str = "index") -> tuple[list[torch.Tensor], list[dict]]:
    index_path = pathlib.Path(index_dir)
    
    if not index_path.exists():
        raise FileNotFoundError(
            f"Index directory '{index_dir}' not found. Run ingest.py first."
        )
    
    # Load all embeddings in order
    npy_files = sorted(index_path.glob("emb_*.npy"), key=lambda p: int(p.stem.split("_")[1]))
    
    if not npy_files:
        raise FileNotFoundError("No embeddings found. Run ingest.py first.")
    
    embeddings = [torch.from_numpy(np.load(f)) for f in npy_files]
    
    with open(index_path / "metadata.json") as f:
        metadata = json.load(f)
    
    print(f"Loaded index: {len(embeddings)} pages")
    return embeddings, metadata