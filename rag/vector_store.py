"""
Vector store wrapper — ChromaDB with OpenAI text-embedding-3-small.

Why ChromaDB
------------
* Works in-memory (Streamlit Cloud) or persistent (local dev).
* Simple Python API; easy to swap for Pinecone / pgvector later.
* Supports metadata filtering (source, page, section).

Why text-embedding-3-small
---------------------------
* 1 536 dimensions, 8 191-token context window.
* Cheap: ~$0.02 / million tokens.
* Strong multilingual performance (important for Copa Airlines' bilingual docs).

Required environment variables
-------------------------------
    OPENAI_API_KEY          – OpenAI API key for embeddings

Optional environment variables
-------------------------------
    CHROMA_PERSIST_DIR      – Local directory for persistent ChromaDB storage.
                              Omit (or set to empty string) to use in-memory mode,
                              which is required on Streamlit Community Cloud.
    CHROMA_COLLECTION       – Collection name (default: copa_knowledge)
"""

from __future__ import annotations

import os
import logging
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

EMBEDDING_MODEL   = "text-embedding-3-small"
DEFAULT_PERSIST   = "./chroma_db"
DEFAULT_COLLECTION = "copa_knowledge"
BATCH_SIZE        = 100  # ChromaDB upsert batch size


def _get_ef() -> embedding_functions.OpenAIEmbeddingFunction:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set")
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=EMBEDDING_MODEL,
    )


class VectorStore:
    """
    Wraps a ChromaDB collection.

    Usage
    -----
    store = VectorStore()
    store.upsert(chunks)                         # index a list of chunk dicts
    results = store.query("how do I sign in?")   # returns top-k chunks
    """

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        # persist_dir=None or "" → in-memory (required on Streamlit Cloud)
        env_dir = os.environ.get("CHROMA_PERSIST_DIR", "").strip()
        self.persist_dir = persist_dir if persist_dir is not None else (env_dir or None)
        self.collection_name = (
            collection_name
            or os.environ.get("CHROMA_COLLECTION", DEFAULT_COLLECTION)
        )
        if self.persist_dir:
            self._client = chromadb.PersistentClient(path=self.persist_dir)
            mode = f"persistent at '{self.persist_dir}'"
        else:
            self._client = chromadb.EphemeralClient()
            mode = "in-memory"
        self._ef = _get_ef()
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "VectorStore ready — collection '%s' (%s, %d docs)",
            self.collection_name,
            mode,
            self._collection.count(),
        )

    @property
    def count(self) -> int:
        return self._collection.count()

    def upsert(self, chunks: list[dict[str, Any]]) -> None:
        """
        Add or update chunks in the vector store.
        Chunks that already exist (by ``chunk_id``) are overwritten.
        """
        if not chunks:
            return
        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start : start + BATCH_SIZE]
            self._collection.upsert(
                ids       = [c["chunk_id"] for c in batch],
                documents = [c["text"]     for c in batch],
                metadatas = [
                    {
                        "source":  c["source"],
                        "page":    c["page"],
                        "section": c.get("section", ""),
                    }
                    for c in batch
                ],
            )
        logger.info("Upserted %d chunks into '%s'", len(chunks), self.collection_name)

    def delete_source(self, source_name: str) -> None:
        """Remove all chunks belonging to a given source document."""
        self._collection.delete(where={"source": source_name})
        logger.info("Deleted all chunks for source '%s'", source_name)

    def query(
        self,
        query_text: str,
        n_results: int = 6,
        source_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Embed *query_text* and return the top-n most similar chunks.

        Returns a list of dicts:
            {
                "chunk_id": str,
                "source":   str,
                "page":     int,
                "section":  str,
                "text":     str,
                "score":    float,   # cosine distance (lower = more similar)
            }
        """
        where = {"source": source_filter} if source_filter else None
        results = self._collection.query(
            query_texts=[query_text],
            n_results=min(n_results, self._collection.count() or 1),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        chunks: list[dict[str, Any]] = []
        ids_list       = results.get("ids",       [[]])[0]
        docs_list      = results.get("documents", [[]])[0]
        metas_list     = results.get("metadatas", [[]])[0]
        distances_list = results.get("distances", [[]])[0]
        for chunk_id, doc, meta, dist in zip(ids_list, docs_list, metas_list, distances_list):
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "source":   meta.get("source", ""),
                    "page":     meta.get("page", 0),
                    "section":  meta.get("section", ""),
                    "text":     doc,
                    "score":    dist,
                }
            )
        return chunks

    def list_sources(self) -> list[str]:
        """Return a sorted list of unique source document names in the store."""
        if self._collection.count() == 0:
            return []
        result = self._collection.get(include=["metadatas"])
        sources = {m.get("source", "") for m in (result.get("metadatas") or [])}
        return sorted(sources)
