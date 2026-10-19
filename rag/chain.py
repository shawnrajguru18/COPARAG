"""
RAG chain — ties retrieval to Claude generation and injects citations.

Flow
----
1. Embed the user question via the VectorStore.
2. Retrieve top-k chunks (cosine similarity).
3. Build a context block containing each chunk's text + citation label.
4. Call Claude with a system prompt that requires citing the labels.
5. Return the assistant reply and the list of citations used.

The context block looks like:

    [DOC-1] Source: guide.pdf | Page 12 | Section: Agent Sign-In
    <text…>

    [DOC-2] Source: guide.pdf | Page 14 | Section: Troubleshooting
    <text…>

Claude is instructed to reference [DOC-N] inline so callers can surface
exact document + page links in the UI.
"""

from __future__ import annotations

import logging
from typing import Any, Generator

import anthropic

from .vector_store import VectorStore

logger = logging.getLogger(__name__)

CLAUDE_MODEL = "claude-opus-4-6"
MAX_TOKENS   = 2_048
TOP_K        = 6

SYSTEM_PROMPT = """You are a precise, expert assistant for Copa Airlines' knowledge base.
You answer questions based exclusively on the reference documents provided below.

## CITATION RULES — mandatory:
- Each reference block is labelled [DOC-N]. You MUST cite the label inline wherever you use information from it, e.g. "…the agent must sign in first [DOC-1]…"
- At the end of every answer, add a ## Sources section listing each cited [DOC-N] with its document name and page.
- If the answer spans multiple references, cite each one where it is used.

## RESPONSE FORMAT RULES — follow strictly:
1. Open with a one-sentence direct answer.
2. Use bold headers for sections (e.g. **Steps to Follow**, **Key Requirements**).
3. For any procedure use a numbered step list.
4. For comparisons use a markdown table.
5. Use callout blocks for warnings/tips:
   > ⚠️ **Warning:** …
   > 💡 **Tip:** …
   > ℹ️ **Note:** …
6. Use exact menu names, button labels, and field names as they appear in the documents.
7. If the information is not in any reference, reply exactly:
   "This topic is not covered in the available knowledge base documents. Please contact your system administrator."
"""


def _build_context(chunks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        header = (
            f"[DOC-{i}] Source: {chunk['source']} | "
            f"Page {chunk['page']}"
            + (f" | Section: {chunk['section']}" if chunk.get("section") else "")
        )
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def stream_answer(
    question: str,
    store: VectorStore,
    conversation: list[dict[str, str]],
    top_k: int = TOP_K,
) -> Generator[str, None, list[dict[str, Any]]]:
    """
    Retrieve relevant chunks, then stream a cited answer from Claude.

    Yields
    ------
    str tokens as they arrive from the API.

    Returns (via StopIteration.value)
    ----------------------------------
    A list of the chunks that were used as context, so the caller can
    render citation cards in the UI.

    Usage
    -----
    gen = stream_answer(question, store, conversation)
    try:
        while True:
            token = next(gen)
            print(token, end="", flush=True)
    except StopIteration as e:
        citations = e.value   # list of chunk dicts
    """
    # 1. Retrieve
    chunks = store.query(question, n_results=top_k)
    if not chunks:
        yield "No relevant documents found. Please ensure documents have been ingested."
        return []

    # 2. Build context
    context = _build_context(chunks)
    system = f"{SYSTEM_PROMPT}\n\n## REFERENCE DOCUMENTS\n\n{context}"

    # 3. Build messages (include prior conversation)
    messages = list(conversation) + [{"role": "user", "content": question}]

    # 4. Stream from Claude
    client = anthropic.Anthropic()
    with client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text

    return chunks


def get_answer(
    question: str,
    store: VectorStore,
    conversation: list[dict[str, str]] | None = None,
    top_k: int = TOP_K,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Non-streaming convenience wrapper. Returns ``(full_text, citations)``.
    """
    conversation = conversation or []
    tokens: list[str] = []
    gen = stream_answer(question, store, conversation, top_k)
    citations: list[dict[str, Any]] = []
    try:
        while True:
            tokens.append(next(gen))
    except StopIteration as exc:
        citations = exc.value or []
    return "".join(tokens), citations
