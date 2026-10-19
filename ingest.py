"""
Ingestion CLI — index documents into the ChromaDB vector store.

Two modes:

  Local mode (no SharePoint credentials needed):
      python ingest.py --local path/to/file.pdf [path/to/more.pdf ...]

  SharePoint mode (requires Azure AD credentials in environment):
      python ingest.py --sharepoint

Run once before starting the Streamlit app, and re-run whenever
documents are updated on SharePoint.

Required environment variables (SharePoint mode only):
    AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET
    SHAREPOINT_SITE_ID, SHAREPOINT_DRIVE_NAME, SHAREPOINT_FOLDER

Required for both modes:
    OPENAI_API_KEY   (for embeddings)

Optional:
    CHROMA_PERSIST_DIR   (default: ./chroma_db)
    CHROMA_COLLECTION    (default: copa_knowledge)
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path

from rag.document_loader import load_document
from rag.chunker import chunk_records
from rag.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def ingest_file(path: Path, store: VectorStore) -> int:
    """Load, chunk, and upsert one file. Returns number of chunks added."""
    logger.info("Processing %s …", path.name)
    try:
        records = load_document(path)
    except Exception as exc:
        logger.error("Failed to load %s: %s", path.name, exc)
        return 0

    chunks = chunk_records(records)
    # Remove stale chunks for this source before upserting fresh ones
    store.delete_source(path.name)
    store.upsert(chunks)
    logger.info("  → %d chunks indexed for %s", len(chunks), path.name)
    return len(chunks)


def ingest_local(paths: list[Path], store: VectorStore) -> None:
    total = 0
    for path in paths:
        total += ingest_file(path, store)
    logger.info("Done. Total chunks indexed: %d", total)
    logger.info("Sources in store: %s", store.list_sources())


def ingest_sharepoint(store: VectorStore) -> None:
    from rag.sharepoint_client import SharePointClient

    client = SharePointClient()
    files = client.list_files()
    logger.info("Found %d files on SharePoint", len(files))

    total = 0
    with tempfile.TemporaryDirectory() as tmp:
        for file_meta in files:
            local_path = client.download_to_file(file_meta, tmp)
            total += ingest_file(local_path, store)

    logger.info("Done. Total chunks indexed: %d", total)
    logger.info("Sources in store: %s", store.list_sources())


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the Copa RAG vector store")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--local",
        nargs="+",
        metavar="FILE",
        help="One or more local PDF/DOCX/PPTX/TXT files to ingest",
    )
    group.add_argument(
        "--sharepoint",
        action="store_true",
        help="Download and ingest all files from the configured SharePoint folder",
    )
    parser.add_argument(
        "--persist-dir",
        default=os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db"),
        help="ChromaDB storage directory (default: ./chroma_db)",
    )
    parser.add_argument(
        "--collection",
        default=os.environ.get("CHROMA_COLLECTION", "copa_knowledge"),
        help="ChromaDB collection name (default: copa_knowledge)",
    )
    args = parser.parse_args()

    store = VectorStore(persist_dir=args.persist_dir, collection_name=args.collection)

    if args.local:
        paths = [Path(p) for p in args.local]
        for p in paths:
            if not p.exists():
                parser.error(f"File not found: {p}")
        ingest_local(paths, store)
    else:
        ingest_sharepoint(store)


if __name__ == "__main__":
    main()
