"""
Ingestion CLI — index documents into the Copa RAG vector store.

Two source modes:

  Local mode (no SharePoint credentials needed):
      python ingest.py --local path/to/file.pdf [path/to/more.pdf ...]

  SharePoint mode (requires Azure AD credentials in environment):
      python ingest.py --sharepoint

Two backend modes:

  ChromaDB (default, local persistent):
      python ingest.py --local docs/ --backend chroma --persist-dir ./chroma_db

  Pinecone (cloud, persistent across Streamlit restarts):
      python ingest.py --local docs/ --backend pinecone
      python ingest.py --local docs/ --backend pinecone --create-index   # first run only

Run once before starting the Streamlit app, and re-run whenever documents change.

Required environment variables (all modes):
    OPENAI_API_KEY            (for embeddings)

Required for Pinecone backend:
    PINECONE_API_KEY
    PINECONE_INDEX_NAME       (optional, default: copa-knowledge)

Required for SharePoint mode:
    AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET
    SHAREPOINT_SITE_ID, SHAREPOINT_DRIVE_NAME, SHAREPOINT_FOLDER
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path

from rag.document_loader import load_document
from rag.chunker import chunk_records
from rag.vector_store import VectorStore, PineconeVectorStore, get_store

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def ingest_file(path: Path, store) -> int:
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


def ingest_local(paths: list[Path], store) -> None:
    total = 0
    for path in paths:
        total += ingest_file(path, store)
    logger.info("Done. Total chunks indexed: %d", total)
    sources = store.list_sources()
    if sources:
        logger.info("Sources in store: %s", sources)


def ingest_sharepoint(store) -> None:
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
    sources = store.list_sources()
    if sources:
        logger.info("Sources in store: %s", sources)


def create_pinecone_index(index_name: str) -> None:
    """Create a Pinecone serverless index if it does not already exist."""
    from pinecone import Pinecone, ServerlessSpec
    from rag.vector_store import EMBEDDING_DIM

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise EnvironmentError("PINECONE_API_KEY is not set")

    pc = Pinecone(api_key=api_key)
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name in existing:
        logger.info("Pinecone index '%s' already exists — skipping creation", index_name)
        return

    logger.info("Creating Pinecone index '%s' (dim=%d, cosine) …", index_name, EMBEDDING_DIM)
    pc.create_index(
        name=index_name,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    logger.info("Index created.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the Copa RAG vector store")

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--local",
        nargs="+",
        metavar="FILE",
        help="One or more local PDF/DOCX/PPTX/TXT files (or directories) to ingest",
    )
    source_group.add_argument(
        "--sharepoint",
        action="store_true",
        help="Download and ingest all files from the configured SharePoint folder",
    )

    parser.add_argument(
        "--backend",
        choices=["chroma", "pinecone"],
        default=None,
        help="Vector store backend (default: pinecone if PINECONE_API_KEY is set, else chroma)",
    )

    # ChromaDB options
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

    # Pinecone options
    parser.add_argument(
        "--index",
        default=os.environ.get("PINECONE_INDEX_NAME", "copa-knowledge"),
        help="Pinecone index name (default: copa-knowledge)",
    )
    parser.add_argument(
        "--create-index",
        action="store_true",
        help="Create the Pinecone index before ingesting (safe to run even if it exists)",
    )

    args = parser.parse_args()

    # Resolve backend
    backend = args.backend or ("pinecone" if os.environ.get("PINECONE_API_KEY") else "chroma")

    if backend == "pinecone":
        if args.create_index:
            create_pinecone_index(args.index)
        os.environ.setdefault("PINECONE_INDEX_NAME", args.index)
        store = PineconeVectorStore(index_name=args.index)
    else:
        store = VectorStore(persist_dir=args.persist_dir, collection_name=args.collection)

    # Expand directories in --local mode
    if args.local:
        paths: list[Path] = []
        for item in args.local:
            p = Path(item)
            if p.is_dir():
                for ext in (".pdf", ".docx", ".pptx", ".txt"):
                    paths.extend(p.rglob(f"*{ext}"))
            else:
                paths.append(p)

        missing = [p for p in paths if not p.exists()]
        if missing:
            parser.error(f"Files not found: {', '.join(str(m) for m in missing)}")

        ingest_local(paths, store)
    else:
        ingest_sharepoint(store)


if __name__ == "__main__":
    main()
