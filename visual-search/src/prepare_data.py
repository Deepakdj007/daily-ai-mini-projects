"""
prepare_data.py
---------------
Downloads the Fashion MNIST dataset and saves a subset of images
to disk as PNG files for indexing.

Run once before indexing:
    uv run python src/prepare_data.py
"""

import os
from pathlib import Path
from datasets import load_dataset
from PIL import Image

# ── Configuration ─────────────────────────────────────────────────
IMAGES_DIR = Path("data/images")
NUM_IMAGES = 500          # subset size — enough to demo, fast to index
SEED = 42                 # reproducible shuffle
# ──────────────────────────────────────────────────────────────────

# Fashion MNIST stores categories as integers 0 through 9.
# This list maps each integer to its human-readable name.
# The order is important — label 0 means tshirt, label 1 means trouser, etc.
LABEL_NAMES = [
    "tshirt",      # 0
    "trouser",     # 1
    "pullover",    # 2
    "dress",       # 3
    "coat",        # 4
    "sandal",      # 5
    "shirt",       # 6
    "sneaker",     # 7
    "bag",         # 8
    "ankle_boot",  # 9
]


def prepare_dataset() -> None:
    # Create the images directory. parents=True creates any missing
    # parent folders. exist_ok=True means no error if folder already exists.
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print("📥 Downloading Fashion MNIST from HuggingFace...")

    # load_dataset streams the data directly from HuggingFace Hub.
    # The first run downloads and caches it locally (~30MB).
    # Every run after that loads instantly from the cache.
    # split="train" gives us the 60,000-image training set to sample from.
    dataset = load_dataset("fashion_mnist", split="train")

    # shuffle() randomizes the order so our 500 images are spread
    # across all 10 categories, not just the first few.
    # select(range(NUM_IMAGES)) then takes the first 500 from the shuffled order.
    dataset = dataset.shuffle(seed=SEED).select(range(NUM_IMAGES))

    print(f"💾 Saving {NUM_IMAGES} images to {IMAGES_DIR}/ ...")

    for idx, item in enumerate(dataset):
        # Each item is a dict with two keys:
        # "image" → a PIL Image object (the actual pixel data)
        # "label" → an integer from 0 to 9 (the category)
        label_name = LABEL_NAMES[item["label"]]

        # Build the filename — e.g. "0042_sneaker.png"
        # {idx:04d} pads the number with leading zeros to 4 digits,
        # so files sort correctly: 0001, 0042, 0123, not 1, 42, 123
        filename = f"{idx:04d}_{label_name}.png"
        filepath = IMAGES_DIR / filename

        img: Image.Image = item["image"]

        # Fashion MNIST images are only 28×28 pixels — very small.
        # CLIP's vision encoder expects 224×224 pixel input.
        # We upscale using LANCZOS, which is the highest quality
        # resizing algorithm in Pillow — it avoids blocky artifacts.
        img = img.resize((224, 224), Image.LANCZOS)

        # Fashion MNIST is grayscale (1 color channel).
        # CLIP was trained on color photos (3 channels: Red, Green, Blue).
        # convert("RGB") copies the grayscale channel into all 3 color slots,
        # making the image structurally valid for CLIP to process.
        img.convert("RGB").save(filepath)

    print(f"✅ Done! {NUM_IMAGES} images saved to {IMAGES_DIR}/")


if __name__ == "__main__":
    prepare_dataset()