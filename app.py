"""
Copa Airlines Knowledge Assistant — RAG edition
Streamlit web interface powered by ChromaDB (in-memory) + OpenAI embeddings + Claude.

Deployment — Streamlit Community Cloud
  1. Push this repo to GitHub.
  2. Connect the repo on share.streamlit.io.
  3. Add secrets in Settings → Secrets (see .streamlit/secrets.toml.example).
  4. Deploy — the app indexes the committed PDF on first load (~30 s cold start).
  5. Drop additional documents (PDFs, Word, PPTX…) via the Upload panel in the sidebar.

Local development
  pip install -r requirements.txt
  cp .streamlit/secrets.toml.example .streamlit/secrets.toml
  streamlit run app.py
"""

from __future__ import annotations

import os
import re
import logging
import tempfile
from pathlib import Path

import streamlit as st

logging.basicConfig(level=logging.INFO)

# ── Documents committed to the repo that are always auto-indexed ──────────────
REPO_DOCUMENTS = [
    "Sales Reporting Solution Guide_v1.2_2026MAR.pdf",
]

SUPPORTED_TYPES = {
    "application/pdf":          ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "text/plain":               ".txt",
}
SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".pptx", ".txt"]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Copa Knowledge Assistant",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Copa brand CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
    --copa-blue:      #003087;
    --copa-blue-dark: #001f5b;
    --copa-gold:      #E8B84B;
    --copa-gold-lt:   #fdf3d9;
    --copa-bg:        #F4F6FA;
    --copa-surface:   #ffffff;
    --copa-border:    #DDE3EF;
    --copa-text:      #1a1a2e;
    --copa-muted:     #6b7280;
    --copa-success:   #10a37f;
}
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
.stApp { background: var(--copa-bg); }

[data-testid="stSidebar"] {
    background: var(--copa-blue) !important;
    border-right: none !important;
}
[data-testid="stSidebar"] * { color: #ffffff !important; }
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stCaption   { color: rgba(255,255,255,0.75) !important; }
[data-testid="stSidebar"] hr           { border-color: rgba(255,255,255,0.2) !important; }
[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    text-align: left !important;
    font-size: 0.82rem !important;
    padding: 6px 10px !important;
    transition: background 0.15s;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(232,184,75,0.25) !important;
    border-color: var(--copa-gold) !important;
}

/* File uploader in sidebar */
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.07) !important;
    border: 1.5px dashed rgba(255,255,255,0.35) !important;
    border-radius: 10px !important;
    padding: 8px !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] * {
    color: rgba(255,255,255,0.85) !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background: var(--copa-gold) !important;
    color: var(--copa-blue-dark) !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 700 !important;
}

.copa-header {
    display: flex; align-items: center; gap: 14px;
    padding: 18px 24px; background: var(--copa-blue);
    border-radius: 12px; margin-bottom: 20px;
    box-shadow: 0 2px 12px rgba(0,48,135,0.18);
}
.copa-header-text h1 { color:#fff; font-size:1.5rem; font-weight:700; margin:0; line-height:1.2; }
.copa-header-text p  { color:var(--copa-gold); font-size:0.82rem; margin:2px 0 0; opacity:.9; }
.copa-badge {
    background: var(--copa-gold); color: var(--copa-blue-dark);
    font-size: 0.68rem; font-weight: 700; padding: 3px 8px;
    border-radius: 20px; letter-spacing: .05em; text-transform: uppercase; white-space: nowrap;
}
.copa-welcome {
    background: var(--copa-surface); border: 1px solid var(--copa-border);
    border-left: 4px solid var(--copa-gold); border-radius: 10px;
    padding: 18px 22px; margin-bottom: 20px;
}
.copa-welcome h3 { color: var(--copa-blue); margin: 0 0 6px; font-size: 1rem; }
.copa-welcome p  { color: var(--copa-muted); margin: 0; font-size: 0.87rem; line-height: 1.5; }
[data-testid="stChatMessage"] {
    border-radius: 12px !important; margin-bottom: 10px !important;
    padding: 14px 18px !important; border: 1px solid var(--copa-border) !important;
    background: var(--copa-surface) !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: var(--copa-gold-lt) !important; border-color: #e8d5a0 !important;
}
.citation-card {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--copa-bg); border: 1px solid var(--copa-border);
    border-radius: 20px; padding: 4px 12px; font-size: 0.78rem;
    color: var(--copa-blue); font-weight: 500; margin: 3px 3px 3px 0; white-space: nowrap;
}
.citation-card .ref-badge {
    background: var(--copa-blue); color: white; border-radius: 10px;
    padding: 1px 6px; font-size: 0.72rem; font-weight: 700;
}
.citation-section {
    margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--copa-border);
}
.citation-section .citation-label {
    font-size: 0.75rem; color: var(--copa-muted); font-weight: 600;
    text-transform: uppercase; letter-spacing: .06em; margin-bottom: 8px;
}
.doc-chip {
    display: flex; align-items: center; gap: 8px;
    background: rgba(16,163,127,0.15); border: 1px solid rgba(16,163,127,0.35);
    border-radius: 8px; padding: 6px 10px; margin-bottom: 5px; font-size: 0.78rem;
}
.upload-chip {
    display: flex; align-items: center; gap: 8px;
    background: rgba(232,184,75,0.18); border: 1px solid rgba(232,184,75,0.4);
    border-radius: 8px; padding: 6px 10px; margin-bottom: 5px; font-size: 0.78rem;
}
[data-testid="stChatInputContainer"] textarea {
    border-radius: 10px !important; border: 1.5px solid var(--copa-border) !important;
}
[data-testid="stChatInputContainer"] textarea:focus {
    border-color: var(--copa-blue) !important;
    box-shadow: 0 0 0 2px rgba(0,48,135,0.12) !important;
}
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Secret helpers ────────────────────────────────────────────────────────────
def _secret(key: str) -> str | None:
    try:
        val = st.secrets.get(key)
        if val:
            return str(val)
    except Exception:
        pass
    return os.environ.get(key)


def _require_secret(key: str, label: str) -> str:
    val = _secret(key)
    if not val:
        st.error(f"**{label}** not found. Add `{key}` in Streamlit Cloud → Settings → Secrets.")
        st.stop()
    return val


# ── Load RAG store (cached for app lifetime) ──────────────────────────────────
@st.cache_resource(show_spinner="✈️  Indexing knowledge base — first load takes ~30 seconds…")
def load_store():
    """Build an in-memory ChromaDB store from committed repo documents."""
    openai_key = _require_secret("OPENAI_API_KEY", "OpenAI API key")
    os.environ["OPENAI_API_KEY"] = openai_key

    from rag.vector_store import VectorStore
    from rag.document_loader import load_document
    from rag.chunker import chunk_records

    store = VectorStore(persist_dir=None)

    repo_root = Path(__file__).parent
    indexed: list[str] = []
    for filename in REPO_DOCUMENTS:
        path = repo_root / filename
        if not path.exists():
            continue
        try:
            store.upsert(chunk_records(load_document(path)))
            indexed.append(filename)
        except Exception as exc:
            st.warning(f"Could not index {filename}: {exc}")

    return store, indexed


def _index_uploaded_file(uploaded_file, store) -> str | None:
    """
    Save an uploaded file to a temp path, index it, return its name on success.
    Returns None on failure (warning already shown).
    """
    from rag.document_loader import load_document
    from rag.chunker import chunk_records

    suffix = Path(uploaded_file.name).suffix.lower()
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = Path(tmp.name)
        # Give the chunk its real filename as source
        records = load_document(tmp_path)
        for r in records:
            r["source"] = uploaded_file.name     # override temp filename
        chunks = chunk_records(records)
        for c in chunks:
            c["source"]   = uploaded_file.name
            c["chunk_id"] = c["chunk_id"].replace(tmp_path.name, uploaded_file.name)
        store.delete_source(uploaded_file.name)  # remove stale version if any
        store.upsert(chunks)
        return uploaded_file.name
    except Exception as exc:
        st.warning(f"Could not index {uploaded_file.name}: {exc}")
        return None
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass


# ── Response rendering ────────────────────────────────────────────────────────
_CALLOUT_RE = re.compile(r'^((?:> .+\n?)+)', re.MULTILINE)


def render_response(text: str) -> None:
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
    if not chunks:
        return
    seen: set[str] = set()
    unique: list[tuple[int, dict]] = []
    for i, c in enumerate(chunks, start=1):
        key = f"{c['source']}|{c['page']}"
        if key not in seen:
            seen.add(key)
            unique.append((i, c))
    pills = ""
    for ref, c in unique:
        section = f" · {c['section']}" if c.get("section") else ""
        pills += (
            f'<span class="citation-card">'
            f'<span class="ref-badge">DOC-{ref}</span>'
            f'📄 {c["source"]} · p.{c["page"]}{section}'
            f'</span>'
        )
    st.markdown(
        f'<div class="citation-section"><div class="citation-label">📎 Sources</div>{pills}</div>',
        unsafe_allow_html=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    try:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/Copa_Airlines_logo.svg/320px-Copa_Airlines_logo.svg.png",
            use_container_width=True,
        )
    except Exception:
        st.markdown("### ✈️ Copa Airlines")
    st.markdown("&nbsp;")

    # ── Knowledge base status ─────────────────────────────────────────────────
    try:
        store, repo_indexed = load_store()

        st.markdown("**Knowledge Base**")
        for f in repo_indexed:
            st.markdown(f'<div class="doc-chip">✅ {f}</div>', unsafe_allow_html=True)

        # Show uploaded docs
        for f in st.session_state.get("uploaded_docs", []):
            st.markdown(f'<div class="upload-chip">📤 {f}</div>', unsafe_allow_html=True)

        st.markdown(
            f"<small style='color:rgba(255,255,255,0.5)'>{store.count} chunks indexed</small>",
            unsafe_allow_html=True,
        )
    except Exception as exc:
        st.error(f"Startup error: {exc}")

    st.divider()

    # ── Document upload ────────────────────────────────────────────────────────
    st.markdown("**📂 Upload Documents**")
    st.caption("PDF, Word, PowerPoint, or TXT — indexed immediately")

    uploaded_files = st.file_uploader(
        "Drop files here",
        type=["pdf", "docx", "pptx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="doc_uploader",
    )

    if uploaded_files:
        if "uploaded_docs" not in st.session_state:
            st.session_state.uploaded_docs = []

        for uf in uploaded_files:
            if uf.name not in st.session_state.uploaded_docs:
                with st.spinner(f"Indexing {uf.name}…"):
                    result = _index_uploaded_file(uf, store)
                if result:
                    st.session_state.uploaded_docs.append(result)
                    st.success(f"✅ Indexed: {uf.name}")
                    st.rerun()

    st.divider()

    # ── Suggested questions ────────────────────────────────────────────────────
    st.markdown("**Suggested Questions**")

    topics = {
        "🔑 Getting Started": [
            "How do I sign in to the Sales Reporting Manager?",
            "What are the hardware and browser requirements?",
            "How do I sign out?",
        ],
        "📊 Reports": [
            "How do I close an Agent Sales Report?",
            "What is the difference between an Agent and a Supervisor report?",
            "How do I view a Station Sales Report?",
        ],
        "💰 Cash Accounting": [
            "How does the Cash Accounting Report work?",
            "How do I balance the Cash Accounting Report?",
            "What is a variance in a Cash Accounting Report?",
        ],
        "🔍 Filters & Transactions": [
            "How do I filter reports by agent?",
            "What is the difference between a void and a refund?",
            "What payment types are supported?",
        ],
    }

    for section, questions in topics.items():
        with st.expander(section, expanded=False):
            for q in questions:
                if st.button(q, key=f"q_{q}", use_container_width=True):
                    st.session_state.pending_prompt = q

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True, key="clear_btn"):
        st.session_state.messages  = []
        st.session_state.citations = {}
        st.rerun()


# ── Main header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="copa-header">
  <div style="font-size:2.6rem;line-height:1">✈️</div>
  <div class="copa-header-text">
    <h1>Copa Airlines — Knowledge Assistant</h1>
    <p>Powered by the Copa Knowledge Base · AI-generated answers with source citations</p>
  </div>
  <div style="margin-left:auto"><span class="copa-badge">RAG Powered</span></div>
</div>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages      = []
if "citations" not in st.session_state:
    st.session_state.citations     = {}
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs  = []

# ── Welcome card ──────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
<div class="copa-welcome">
  <h3>👋 Welcome to the Copa Knowledge Assistant</h3>
  <p>
    Ask any question about Copa's systems — sign-in procedures, report types, cash accounting,
    transaction processing, and more. Every answer is drawn directly from official Copa documents
    and includes the exact page and section it came from.<br><br>
    Use the <strong>Suggested Questions</strong> in the sidebar to get started, or
    <strong>upload additional documents</strong> (PDFs, Word files, PowerPoint decks) from your
    SharePoint folder — they're indexed and searchable immediately.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Chat history ──────────────────────────────────────────────────────────────
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_response(msg["content"])
            render_citations(st.session_state.citations.get(idx, []))
        else:
            st.markdown(msg["content"])

# ── Prompt input ──────────────────────────────────────────────────────────────
active_prompt = None
if st.session_state.pending_prompt:
    active_prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

typed_prompt = st.chat_input("Ask anything about Copa's systems…")
prompt = typed_prompt or active_prompt

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history = st.session_state.messages[:-1][-10:]

    with st.chat_message("assistant"):
        stream_placeholder = st.empty()
        full_response      = ""
        retrieved_chunks: list[dict] = []

        try:
            store, _ = load_store()
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
            full_response = f"⚠️ An error occurred: {exc}"

        stream_placeholder.empty()
        render_response(full_response)
        render_citations(retrieved_chunks)

    msg_idx = len(st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.citations[msg_idx] = retrieved_chunks
