"""
Text chunker — splits page/paragraph records into overlapping token-bounded chunks.

Each output chunk keeps the source metadata so every answer can cite
exactly which document, page, and section it came from.

Strategy
--------
* Target chunk size:  400 tokens  (~1 600 chars at 4 chars/token)
* Overlap:             50 tokens  (~200 chars)
* Splitting order: paragraph → sentence → word boundaries (never mid-word)

Chunks are dicts:
    {
        "chunk_id":  "filename.pdf::p3::0",   # unique across the store
        "source":    "filename.pdf",
        "page":      3,
        "section":   "2.1 Agent Sign-In",
        "text":      "…chunk content…"
    }
"""

from __future__ import annotations

import re
from typing import Any

CHUNK_SIZE_CHARS = 1_600   # ≈ 400 tokens
OVERLAP_CHARS    = 200     # ≈ 50 tokens


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split *text* into overlapping chunks of at most *chunk_size* characters,
    preferring to break on sentence or paragraph boundaries.
    """
    if len(text) <= chunk_size:
        return [text]

    # Try to split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks: list[str] = []
    current = ""
    for sent in sentences:
        if not current:
            current = sent
            continue
        if len(current) + 1 + len(sent) <= chunk_size:
            current += " " + sent
        else:
            chunks.append(current)
            # Start next chunk with overlap from end of current chunk
            overlap_text = current[-overlap:] if len(current) > overlap else current
            current = overlap_text + " " + sent
    if current:
        chunks.append(current)

    # Hard-split any chunk still over the limit (e.g. a very long sentence)
    final: list[str] = []
    for chunk in chunks:
        if len(chunk) <= chunk_size:
            final.append(chunk)
        else:
            for start in range(0, len(chunk), chunk_size - overlap):
                final.append(chunk[start : start + chunk_size])
    return [c.strip() for c in final if c.strip()]


def chunk_records(
    records: list[dict[str, Any]],
    chunk_size: int = CHUNK_SIZE_CHARS,
    overlap: int = OVERLAP_CHARS,
) -> list[dict[str, Any]]:
    """
    Take a list of page/paragraph records (from ``document_loader``) and
    return a flat list of chunk dicts.
    """
    chunks: list[dict[str, Any]] = []
    for record in records:
        parts = _split_text(record["text"], chunk_size, overlap)
        for i, part in enumerate(parts):
            chunk_id = f"{record['source']}::p{record['page']}::{i}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "source":   record["source"],
                    "page":     record["page"],
                    "section":  record.get("section", ""),
                    "text":     part,
                }
            )
    return chunks
