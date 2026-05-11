"""
indexer.py
----------
Offline script: reads all product images, encodes them with CLIP,
and stores the vectors in Qdrant.

Run once after prepare_data.py:
    uv run python src/indexer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure our src/ modules are importable when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedder import CLIPEmbedder
from src.store import VectorStore

# The folder where prepare_data.py saved the product images
IMAGES_DIR = Path("data/images")

# Number of images to encode before writing to Qdrant.
# 32 is a good default: enough to amortize disk write overhead
# without holding too many vectors in RAM at once.
BATCH_SIZE = 32

def extract_category(filename: str) -> str:
    """
    Parse the category label from our filename naming convention.

    Our prepare_data.py saves files as: "{index}_{category}.png"
    Examples:
        "0042_sneaker.png"    → "sneaker"
        "0001_dress.png"      → "dress"
        "0200_ankle_boot.png" → "ankle_boot"
    """
    # .stem strips the file extension: "0042_sneaker.png" → "0042_sneaker"
    stem = Path(filename).stem

    # Split on the FIRST underscore only (maxsplit=1).
    # This keeps compound category names like "ankle_boot" together:
    # "0200_ankle_boot".split("_", maxsplit=1) → ["0200", "ankle_boot"]
    parts = stem.split("_", maxsplit=1)

    return parts[1] if len(parts) > 1 else "unknown"

def index_images() -> None:
    """Encode all images with CLIP and store their vectors in Qdrant."""

    # sorted() ensures consistent alphabetical processing order
    image_paths = sorted(IMAGES_DIR.glob("*.png"))

    if not image_paths:
        # No images found — user probably forgot to run prepare_data.py
        print(f"❌ No images found in {IMAGES_DIR}/")
        print("   Run: uv run python src/prepare_data.py")
        sys.exit(1)

    print(f"🖼️  Found {len(image_paths)} images to index")

    embedder = CLIPEmbedder()   # loads CLIP (~5 seconds, ~350MB)
    store = VectorStore()        # connects to or creates Qdrant collection

    # If we already indexed all images in a previous run, skip.
    # The user can delete data/qdrant_db/ to force a full re-index
    # (useful when adding new images to the catalog).
    if store.count() >= len(image_paths):
        print(f"✅ Already indexed {store.count()} images. Nothing to do.")
        print("   Delete data/qdrant_db/ to force a re-index.")
        return
    
    # We collect images into these three parallel lists.
    # They grow together — entry N in each list belongs to the same image.
    # When the batch reaches BATCH_SIZE, we flush all three to Qdrant.
    batch_ids: list[int] = []
    batch_vectors = []
    batch_payloads: list[dict] = []
    total_indexed = 0

    for idx, image_path in enumerate(image_paths):

        # Encode this image to a 512-dim normalized vector.
        # try/except means one corrupt or unreadable image
        # does not crash the entire indexing run.
        try:
            vector = embedder.embed_image(image_path)
        except Exception as e:
            print(f"⚠️  Skipping {image_path.name}: {e}")
            continue

        # The payload is metadata stored alongside the vector in Qdrant.
        # When search returns a result, we get this payload back —
        # it tells us which file the matching vector came from.
        payload = {
            "image_path": str(image_path),           # full path to open the image
            "filename": image_path.name,              # just the filename for display
            "category": extract_category(image_path.name),  # e.g. "sneaker"
        }

        batch_ids.append(idx)
        batch_vectors.append(vector)
        batch_payloads.append(payload)

        # When the batch is full, flush it to Qdrant and clear the lists
        if len(batch_ids) == BATCH_SIZE:
            store.upsert_batch(batch_ids, batch_vectors, batch_payloads)
            total_indexed += len(batch_ids)
            print(f"   Indexed {total_indexed}/{len(image_paths)} images...")
            batch_ids.clear()
            batch_vectors.clear()
            batch_payloads.clear()

    # After the loop ends, the last batch may not have been full.
    # Example: 500 images ÷ 32 per batch = 15 full batches (480 images)
    # with 20 leftover. This final block flushes those 20.
    if batch_ids:
        store.upsert_batch(batch_ids, batch_vectors, batch_payloads)
        total_indexed += len(batch_ids)

    print(f"\n✅ Indexing complete! {total_indexed} images stored in Qdrant.")


if __name__ == "__main__":
    index_images()