"""
embedder.py
-----------
Wraps the CLIP model with two methods:
  - embed_image(path)  → 512-dim vector from an image file
  - embed_text(query)  → 512-dim vector from a text string

Both vectors live in the same mathematical space — that is what makes
text-to-image similarity search possible.
"""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# The CLIP model we are using. Let's decode the name:
#   openai     → released by OpenAI, available on HuggingFace
#   clip       → Contrastive Language-Image Pre-training
#   vit        → Vision Transformer (the image encoder architecture)
#   base       → medium-sized model, good balance of speed and quality
#   patch32    → the image is divided into 32×32 pixel patches for the ViT
#
# This model outputs 512-dimensional embeddings.
# Upgrade option: "openai/clip-vit-large-patch14" → 768-dim, more accurate, slower
MODEL_ID = "openai/clip-vit-base-patch32"

class CLIPEmbedder:
    """
    Loads CLIP once and exposes embed_image / embed_text methods.

    Always create one instance and reuse it. Loading the model takes
    ~5 seconds and ~350MB of RAM — you do not want to do this per request.
    """

    def __init__(self) -> None:
        # Check if a GPU is available. GPU makes CLIP run 10-20x faster.
        # If there is no GPU, CPU works fine — just slower for large datasets.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Loading CLIP on: {self.device}")

        # CLIPProcessor is the preprocessing pipeline for both images and text.
        # For images: it resizes to 224×224, normalizes pixel values, converts to tensor.
        # For text: it tokenizes words, pads/truncates to max length, converts to tensor.
        # It handles all the formatting that the CLIP model expects as input.
        self.processor = CLIPProcessor.from_pretrained(MODEL_ID)

        # CLIPModel contains two encoders inside it:
        #   self.model.get_image_features() → runs only the vision encoder (ViT)
        #   self.model.get_text_features()  → runs only the text encoder (Transformer)
        # Both encoders share the same embedding space — that is what enables
        # text-to-image similarity.
        self.model = CLIPModel.from_pretrained(MODEL_ID).to(self.device)

        # eval() switches the model from training mode to inference mode.
        # In training mode, Dropout randomly zeroes out some neurons on each
        # forward pass — this is a regularization technique during training.
        # In inference mode (eval), Dropout is disabled so results are deterministic.
        self.model.eval()

        print("✅ CLIP model ready.")
    
    def embed_image(self, image_path: str | Path) -> np.ndarray:
        """
        Encode one image file into a 512-dimensional vector.

        Args:
            image_path: Path to any PNG or JPEG file.

        Returns:
            numpy array of shape (512,), normalized to unit length.
        """
        # Open the image and ensure it has 3 color channels (RGB).
        # Some images are RGBA (4 channels with transparency) or
        # grayscale (1 channel). CLIP's vision encoder requires exactly 3.
        image = Image.open(image_path).convert("RGB")

        # The processor handles all preprocessing CLIP needs:
        # resizes to 224×224, normalizes pixel values from [0,255] to [-1,1],
        # and converts to a PyTorch tensor.
        # return_tensors="pt" means "return PyTorch tensors" (as opposed to numpy or TF)
        inputs = self.processor(images=image, return_tensors="pt")

        # Move tensors to the same device as the model.
        # If the model is on GPU but the input tensors are on CPU,
        # PyTorch will raise a RuntimeError. This line keeps them in sync.
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # torch.no_grad() tells PyTorch not to track gradients during this block.
        # During training, PyTorch stores gradients for every operation so it can
        # run backpropagation later. During inference, we never backpropagate,
        # so tracking gradients wastes both memory and time.
        with torch.no_grad():
            # get_image_features() runs the ViT vision encoder and returns
            # the image embedding. It skips the text encoder entirely.
            # The ** unpacks our inputs dict into keyword arguments.
            # Output shape: (1, 512) — a batch of 1 vector with 512 numbers
            output = self.model.get_image_features(**inputs)
            features = output if isinstance(output, torch.Tensor) else output[1]

            # === Why we normalize ===
            # Raw CLIP vectors have different magnitudes (lengths).
            # Cosine similarity measures only the *angle* between vectors,
            # not their length. But if vectors have different lengths,
            # a longer vector will artificially score higher even if it
            # points in a slightly wrong direction.
            #
            # Dividing each vector by its own L2 norm (its length) places
            # every vector on the surface of a unit sphere — every vector
            # has exactly length 1.0. Now cosine similarity purely measures
            # the angle, which is exactly what we want.
            #
            # dim=-1 means normalize along the last dimension (the 512 numbers).
            # keepdim=True keeps the shape as (1, 512) instead of collapsing it.
            features = features / features.norm(dim=-1, keepdim=True)

            # .squeeze() removes the batch dimension: (1, 512) → (512,)
            # .cpu() moves to CPU memory (safe even if already on CPU)
            # .numpy() converts the PyTorch tensor to a numpy array for Qdrant
            return features.squeeze().cpu().numpy()
    def embed_text(self, query: str) -> np.ndarray:
        """
        Encode a text query into a 512-dimensional vector.

        Args:
            query: Natural language description, e.g. "red sneakers".

        Returns:
            numpy array of shape (512,), normalized to unit length.
        """
        # The processor tokenizes the text into subword tokens,
        # converts each token to an integer ID, and pads/truncates
        # to CLIP's maximum sequence length of 77 tokens.
        # text=[query] is a list because the processor expects a batch —
        # even if the batch only contains one string.
        inputs = self.processor(
            text=[query],
            return_tensors="pt",
            padding=True,
            truncation=True,   # queries longer than 77 tokens are silently truncated
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            # get_text_features() runs the Transformer text encoder.
            # It skips the vision encoder — we only need the text side here.
            # Output shape: (1, 512)
            output = self.model.get_text_features(**inputs)
            features = output if isinstance(output, torch.Tensor) else output[1]

        # Same normalization as embed_image().
        # This is critical — both image and text vectors must be normalized
        # the same way for the distance between them to be meaningful.
        features = features / features.norm(dim=-1, keepdim=True)

        return features.squeeze().cpu().numpy()