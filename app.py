"""
Copa Airlines Knowledge Chatbot — RAG edition
Streamlit web interface powered by ChromaDB + OpenAI embeddings + Claude.

Before running:
    1. Set environment variables (see .env.example).
    2. Ingest documents once:
           python ingest.py --local "Sales Reporting Solution Guide_v1.2_2026MAR.pdf"
       or for SharePoint:
           python ingest.py --sharepoint
    3. Launch:
           streamlit run app.py
"""

from __future__ import annotations

import os
import re
import logging

import streamlit as st

logging.basicConfig(level=logging.INFO)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Copa Airlines | Knowledge Assistant",
    page_icon="✈️",
    layout="wide",
)


# ── Load RAG components (cached for the app lifetime) ─────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base…")
def load_store():
    """Initialise the ChromaDB vector store (no re-embedding on reload)."""
    from rag.vector_store import VectorStore
    return VectorStore()


@st.cache_resource
def get_anthropic_client():
    import anthropic
    api_key = st.secrets.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        st.error("ANTHROPIC_API_KEY not found. Add it in Streamlit Secrets or as an env var.")
        st.stop()
    return anthropic.Anthropic(api_key=api_key)


# ── Response rendering ────────────────────────────────────────────────────────
_CALLOUT_RE = re.compile(r'^((?:> .+\n?)+)', re.MULTILINE)


def render_response(text: str) -> None:
    """Render markdown with enriched callout blocks (⚠️ / 💡 / ℹ️)."""
    for part in _CALLOUT_RE.split(text):
        if not part.strip():
            continue
        if part.startswith(">"):
            lines = [ln.lstrip("> ").rstrip() for ln in part.strip().splitlines()]
            block = "\n".join(lines)
            first = block.lstrip()
            if first.startswith("⚠️"):
                st.warning(block)
            elif first.startswith("💡"):
                st.success(block)
            else:
                st.info(block)
        else:
            st.markdown(part)


def render_citations(chunks: list[dict]) -> None:
    """Render a compact citation table below the answer."""
    if not chunks:
        return
    with st.expander("📎 Sources used", expanded=False):
        rows = []
        seen: set[str] = set()
        for i, c in enumerate(chunks, start=1):
            key = f"{c['source']}|{c['page']}"
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "Ref": f"[DOC-{i}]",
                    "Document": c["source"],
                    "Page": c["page"],
                    "Section": c.get("section", "—") or "—",
                }
            )
        st.table(rows)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    try:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/Copa_Airlines_logo.svg/320px-Copa_Airlines_logo.svg.png",
            use_container_width=True,
        )
    except Exception:
        st.markdown("## ✈️ Copa Airlines")

    st.markdown("## Knowledge Assistant")
    st.caption("RAG-powered · Cites sources · Multi-document")
    st.divider()

    # Store status
    try:
        store = load_store()
        sources = store.list_sources()
        st.markdown(f"**Indexed documents:** {len(sources)}")
        if sources:
            for src in sources:
                st.caption(f"• {src}")
        else:
            st.warning(
                "No documents indexed yet. Run:\n\n"
                "```\npython ingest.py --local your_file.pdf\n```"
            )
    except Exception as exc:
        st.error(f"Vector store error: {exc}")

    st.divider()
    st.markdown("### Suggested Questions")

    topics = {
        "Getting Started": [
            "How do I sign in to the Sales Reporting Manager?",
            "What are the hardware and browser requirements?",
        ],
        "Reports": [
            "How do I close an Agent Sales Report?",
            "What is the difference between an Agent and a Supervisor report?",
        ],
        "Cash Accounting": [
            "How does the Cash Accounting Report work?",
            "How do I balance the Cash Accounting Report?",
        ],
        "Transactions": [
            "What is the difference between a void and a refund in the Sales Report?",
            "What payment types are supported?",
        ],
    }

    for section, questions in topics.items():
        with st.expander(section):
            for q in questions:
                if st.button(q, key=q, use_container_width=True):
                    st.session_state.pending_prompt = q

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("✈️ Copa Airlines — Knowledge Assistant")
st.caption("Ask anything about Copa's systems. Answers cite exact document pages.")

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "citations" not in st.session_state:
    st.session_state.citations = {}    # message index → list of chunk dicts
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# ── Render chat history ───────────────────────────────────────────────────────
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_response(msg["content"])
            render_citations(st.session_state.citations.get(idx, []))
        else:
            st.markdown(msg["content"])

# ── Resolve sidebar button as prompt ─────────────────────────────────────────
active_prompt = None
if st.session_state.pending_prompt:
    active_prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

typed_prompt = st.chat_input("Ask a question about Copa's knowledge base…")
prompt = typed_prompt or active_prompt

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build conversation history for context (last 10 turns)
    history = st.session_state.messages[:-1][-10:]

    with st.chat_message("assistant"):
        stream_placeholder = st.empty()
        full_response = ""
        retrieved_chunks: list[dict] = []

        try:
            store = load_store()
            from rag.chain import stream_answer

            gen = stream_answer(prompt, store, history)
            try:
                while True:
                    token = next(gen)
                    full_response += token
                    stream_placeholder.markdown(full_response + "▌")
            except StopIteration as exc:
                retrieved_chunks = exc.value or []

        except Exception as exc:
            full_response = f"⚠️ Error: {exc}"

        stream_placeholder.empty()
        render_response(full_response)
        render_citations(retrieved_chunks)

    msg_idx = len(st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.citations[msg_idx] = retrieved_chunks
