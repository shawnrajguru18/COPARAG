"""
Vector store — two backends:

  ChromaDB (default)
  ------------------
  In-memory (EphemeralClient) or persistent (PersistentClient).
  Good for: local dev, committed sample docs, browser uploads.
  Required env: OPENAI_API_KEY

  Pinecone
  --------
  Persistent cloud vector store; survives Streamlit restarts; handles 300 MB+.
  Index once locally with ingest.py, query forever from the cloud.
  Required env: OPENAI_API_KEY, PINECONE_API_KEY
  Optional env: PINECONE_INDEX_NAME (default: copa-knowledge)

Factory
-------
    from rag.vector_store import get_store
    store = get_store()   # auto-detects: Pinecone if key present, else ChromaDB
"""

from __future__ import annotations

import os
import logging
from typing import Any

logger = logging.getLogger(__name__)

EMBEDDING_MODEL    = "text-embedding-3-small"
EMBEDDING_DIM      = 1536
BATCH_SIZE         = 100

# ── ChromaDB ──────────────────────────────────────────────────────────────────

DEFAULT_PERSIST    = "./chroma_db"
DEFAULT_COLLECTION = "copa_knowledge"


def _get_chroma_ef():
    from chromadb.utils import embedding_functions
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set")
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=EMBEDDING_MODEL,
    )


class VectorStore:
    """
    ChromaDB-backed vector store.

    persist_dir=None → in-memory (required on Streamlit Community Cloud).
    """

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        import chromadb
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
        self._ef = _get_chroma_ef()
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "VectorStore ready — collection '%s' (%s, %d docs)",
            self.collection_name, mode, self._collection.count(),
        )

    @property
    def count(self) -> int:
        return self._collection.count()

    def upsert(self, chunks: list[dict[str, Any]]) -> None:
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
        self._collection.delete(where={"source": source_name})
        logger.info("Deleted all chunks for source '%s'", source_name)

    def query(
        self,
        query_text: str,
        n_results: int = 6,
        source_filter: str | None = None,
    ) -> list[dict[str, Any]]:
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
            chunks.append({
                "chunk_id": chunk_id,
                "source":   meta.get("source", ""),
                "page":     meta.get("page", 0),
                "section":  meta.get("section", ""),
                "text":     doc,
                "score":    dist,
            })
        return chunks

    def list_sources(self) -> list[str]:
        if self._collection.count() == 0:
            return []
        result = self._collection.get(include=["metadatas"])
        sources = {m.get("source", "") for m in (result.get("metadatas") or [])}
        return sorted(sources)


# ── Pinecone ──────────────────────────────────────────────────────────────────

DEFAULT_PINECONE_INDEX = "copa-knowledge"


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts with OpenAI text-embedding-3-small."""
    import openai
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


class PineconeVectorStore:
    """
    Pinecone-backed vector store.

    Index must already exist (run  ingest.py --backend pinecone --create-index
    once to provision it, then ingest your documents).

    Required env vars: OPENAI_API_KEY, PINECONE_API_KEY
    Optional env vars: PINECONE_INDEX_NAME  (default: copa-knowledge)
    """

    def __init__(self, index_name: str | None = None) -> None:
        from pinecone import Pinecone

        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise EnvironmentError("PINECONE_API_KEY is not set")
        if not os.environ.get("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY is not set")

        self._index_name = (
            index_name
            or os.environ.get("PINECONE_INDEX_NAME", DEFAULT_PINECONE_INDEX)
        )
        pc = Pinecone(api_key=api_key)
        self._index = pc.Index(self._index_name)
        stats = self._index.describe_index_stats()
        logger.info(
            "PineconeVectorStore ready — index '%s' (%d vectors)",
            self._index_name, stats.total_vector_count,
        )

    @property
    def count(self) -> int:
        return self._index.describe_index_stats().total_vector_count

    def upsert(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            return
        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start : start + BATCH_SIZE]
            texts = [c["text"] for c in batch]
            embeddings = _embed_texts(texts)
            vectors = [
                {
                    "id": c["chunk_id"],
                    "values": emb,
                    "metadata": {
                        "source":  c["source"],
                        "page":    int(c["page"]),
                        "section": c.get("section", ""),
                        "text":    c["text"],
                    },
                }
                for c, emb in zip(batch, embeddings)
            ]
            self._index.upsert(vectors=vectors)
        logger.info("Upserted %d chunks into Pinecone index '%s'", len(chunks), self._index_name)

    def delete_source(self, source_name: str) -> None:
        self._index.delete(filter={"source": {"$eq": source_name}})
        logger.info("Deleted all chunks for source '%s' from Pinecone", source_name)

    def query(
        self,
        query_text: str,
        n_results: int = 6,
        source_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        embedding = _embed_texts([query_text])[0]
        filter_ = {"source": {"$eq": source_filter}} if source_filter else None
        results = self._index.query(
            vector=embedding,
            top_k=n_results,
            filter=filter_,
            include_metadata=True,
        )
        chunks: list[dict[str, Any]] = []
        for match in results.matches:
            meta = match.metadata or {}
            chunks.append({
                "chunk_id": match.id,
                "source":   meta.get("source", ""),
                "page":     meta.get("page", 0),
                "section":  meta.get("section", ""),
                "text":     meta.get("text", ""),
                "score":    1.0 - match.score,  # convert similarity → distance for consistency
            })
        return chunks

    def list_sources(self) -> list[str]:
        # Pinecone doesn't support distinct metadata aggregation; return empty list.
        # Sources are tracked in-app via session_state instead.
        return []


# ── Factory ───────────────────────────────────────────────────────────────────

def get_store(
    backend: str | None = None,
    persist_dir: str | None = None,
    collection_name: str | None = None,
    pinecone_index: str | None = None,
) -> VectorStore | PineconeVectorStore:
    """
    Return the appropriate vector store.

    backend=None → auto-detect: Pinecone if PINECONE_API_KEY is set, else ChromaDB.
    backend="pinecone"  → PineconeVectorStore
    backend="chroma"    → VectorStore (ChromaDB)
    """
    resolved = backend or (
        "pinecone" if os.environ.get("PINECONE_API_KEY") else "chroma"
    )
    if resolved == "pinecone":
        return PineconeVectorStore(index_name=pinecone_index)
    return VectorStore(persist_dir=persist_dir, collection_name=collection_name)
