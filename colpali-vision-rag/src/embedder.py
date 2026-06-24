"""ColPali (ColQwen2) visual embedder — the model that "reads" pages by sight.

It turns each page image into a *multivector*: one 128-d vector per image patch,
not a single vector for the whole page. Text queries get the same treatment
(one vector per token). Matching them with MaxSim is what makes the retrieval
sensitive to small regions like a single bar in a chart or a cell in a table.

The model is heavy, so we load it once and reuse it (lazy singleton).
"""

import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor
from PIL import Image

from src.config import COLPALI_MODEL

_model: ColQwen2 | None = None
_processor: ColQwen2Processor | None = None


def _device_and_dtype() -> tuple[str, torch.dtype]:
    """Pick CUDA + bfloat16 when a GPU is present, else CPU + float32."""
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.float32


def load_model() -> tuple[ColQwen2, ColQwen2Processor]:
    """Load the ColQwen2 model and processor once, then cache them."""
    global _model, _processor
    if _model is not None and _processor is not None:
        return _model, _processor

    device, dtype = _device_and_dtype()
    _model = ColQwen2.from_pretrained(
        COLPALI_MODEL,
        torch_dtype=dtype,
        device_map=device,
    ).eval()
    _processor = ColQwen2Processor.from_pretrained(COLPALI_MODEL)
    print(f"Loaded {COLPALI_MODEL} on {device} ({dtype})")
    return _model, _processor


def _to_multivector(embedding: torch.Tensor) -> list[list[float]]:
    """Convert one (n_patches, 128) tensor into plain Python floats for Qdrant."""
    return embedding.to(torch.float32).cpu().numpy().tolist()


def embed_image(image: Image.Image) -> list[list[float]]:
    """Embed a single page image into its patch-level multivector.

    We process one image at a time so each page's multivector contains only its
    own real patches — no batch padding sneaks in as junk vectors.
    """
    model, processor = load_model()
    inputs = processor.process_images([image]).to(model.device)
    with torch.no_grad():
        out = model(**inputs)  # shape: (1, n_patches, 128)
    return _to_multivector(out[0])


def embed_query(text: str) -> list[list[float]]:
    """Embed a text query into its token-level multivector."""
    model, processor = load_model()
    inputs = processor.process_queries([text]).to(model.device)
    with torch.no_grad():
        out = model(**inputs)  # shape: (1, n_tokens, 128)
    return _to_multivector(out[0])
