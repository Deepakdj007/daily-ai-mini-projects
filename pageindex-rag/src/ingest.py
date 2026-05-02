import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from pageindex import PageIndexClient

load_dotenv()

# Initialise the client once — it reads PAGEINDEX_API_KEY from env
client = PageIndexClient(api_key=os.getenv("PAGEINDEX_API_KEY"))

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def upload_document(pdf_path: str) -> str:
    """
    Upload a PDF to PageIndex and return the doc_id.
    Saves the doc_id locally so you don't re-upload the same file.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"Uploading {pdf_path.name}...")
    result = client.submit_document(str(pdf_path))
    doc_id = result["doc_id"]

    # Persist locally — we never want to lose a doc_id
    record = {"doc_id": doc_id, "filename": pdf_path.name}
    record_path = DATA_DIR / f"{pdf_path.stem}.json"
    record_path.write_text(json.dumps(record, indent=2))

    print(f"Document submitted. doc_id: {doc_id}")
    return doc_id

def wait_for_processing(doc_id: str, poll_interval: int = 5, timeout: int = 300) -> dict:
    """
    Poll PageIndex until the document finishes processing.
    Returns the full tree structure when ready.
    """
    print(f"Waiting for document {doc_id} to process...")
    start = time.time()

    while True:
        elapsed = time.time() - start
        if elapsed > timeout:
            raise TimeoutError(f"Document still processing after {timeout}s. Try again later.")

        result = client.get_tree(doc_id)
        status = result.get("status")

        if status == "completed":
            print(f"Processing complete in {elapsed:.0f}s")
            return result

        if status == "failed":
            raise RuntimeError(f"PageIndex failed to process document: {result}")

        print(f"  Status: {status} — checking again in {poll_interval}s...")
        time.sleep(poll_interval)

def ingest(pdf_path: str) -> dict:
    """Full pipeline: upload, wait, return the tree."""
    doc_id = upload_document(pdf_path)
    tree = wait_for_processing(doc_id)

    # Save the tree locally for inspection
    tree_path = DATA_DIR / f"{Path(pdf_path).stem}_tree.json"
    tree_path.write_text(json.dumps(tree, indent=2))
    print(f"Tree saved to {tree_path}")

    return tree

def load_doc_id(filename_stem: str) -> str:
    """
    Load a previously saved doc_id by the PDF filename (without extension).
    Example: load_doc_id('annual-report') loads data/annual-report.json
    """
    record_path = DATA_DIR / f"{filename_stem}.json"
    if not record_path.exists():
        raise FileNotFoundError(
            f"No saved doc_id for '{filename_stem}'. Run ingest() first."
        )
    return json.loads(record_path.read_text())["doc_id"]