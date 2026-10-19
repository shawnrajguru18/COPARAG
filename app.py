"""
Copa Airlines Knowledge Assistant — RAG edition
Streamlit web interface powered by ChromaDB (in-memory) + OpenAI embeddings + Claude.

Deployment — Streamlit Community Cloud
  1. Push this repo to GitHub.
  2. Connect the repo on share.streamlit.io.
  3. Add secrets in Settings → Secrets (see .streamlit/secrets.toml.example).
  4. Deploy — the app indexes documents on first load (~30 s cold start).

Local development
  pip install -r requirements.txt
  cp .streamlit/secrets.toml.example .streamlit/secrets.toml
  # fill in ANTHROPIC_API_KEY and OPENAI_API_KEY
  streamlit run app.py
"""

from __future__ import annotations

import os
import re
import logging
from pathlib import Path

import streamlit as st

logging.basicConfig(level=logging.INFO)

# ── Documents to auto-index on startup ────────────────────────────────────────
# Add filenames here as you commit new documents to the repo.
REPO_DOCUMENTS = [
    "Sales Reporting Solution Guide_v1.2_2026MAR.pdf",
]

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
/* ── Brand tokens ───────────────────────────────────────────────────────── */
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
    --copa-warn:      #f59e0b;
    --copa-error:     #ef4444;
}

/* ── Global ─────────────────────────────────────────────────────────────── */
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

.stApp { background: var(--copa-bg); }

/* ── Sidebar ────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--copa-blue) !important;
    border-right: none !important;
}
[data-testid="stSidebar"] * { color: #ffffff !important; }
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stCaption   { color: rgba(255,255,255,0.75) !important; }
[data-testid="stSidebar"] hr           { border-color: rgba(255,255,255,0.2) !important; }

/* Sidebar buttons (suggested questions) */
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
/* Clear button */
[data-testid="stSidebar"] .stButton:last-of-type > button {
    background: rgba(239,68,68,0.15) !important;
    border-color: rgba(239,68,68,0.4) !important;
}
[data-testid="stSidebar"] .stButton:last-of-type > button:hover {
    background: rgba(239,68,68,0.3) !important;
}

/* ── Main header ────────────────────────────────────────────────────────── */
.copa-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 18px 24px;
    background: var(--copa-blue);
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 2px 12px rgba(0,48,135,0.18);
}
.copa-header-text h1 {
    color: #ffffff;
    font-size: 1.5rem;
    font-weight: 700;
    margin: 0;
    line-height: 1.2;
}
.copa-header-text p {
    color: var(--copa-gold);
    font-size: 0.82rem;
    margin: 2px 0 0;
    opacity: 0.9;
}
.copa-badge {
    background: var(--copa-gold);
    color: var(--copa-blue-dark);
    font-size: 0.68rem;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 20px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    white-space: nowrap;
}

/* ── Welcome card ────────────────────────────────────────────────────────── */
.copa-welcome {
    background: var(--copa-surface);
    border: 1px solid var(--copa-border);
    border-left: 4px solid var(--copa-gold);
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 20px;
}
.copa-welcome h3 { color: var(--copa-blue); margin: 0 0 6px; font-size: 1rem; }
.copa-welcome p  { color: var(--copa-muted); margin: 0; font-size: 0.87rem; line-height: 1.5; }

/* ── Chat messages ───────────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    border-radius: 12px !important;
    margin-bottom: 10px !important;
    padding: 14px 18px !important;
    border: 1px solid var(--copa-border) !important;
    background: var(--copa-surface) !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
}
/* User messages */
[data-testid="stChatMessage"][data-testid*="user"],
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: var(--copa-gold-lt) !important;
    border-color: #e8d5a0 !important;
}

/* ── Citation cards ─────────────────────────────────────────────────────── */
.citation-card {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--copa-bg);
    border: 1px solid var(--copa-border);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.78rem;
    color: var(--copa-blue);
    font-weight: 500;
    margin: 3px 3px 3px 0;
    white-space: nowrap;
}
.citation-card .ref-badge {
    background: var(--copa-blue);
    color: white;
    border-radius: 10px;
    padding: 1px 6px;
    font-size: 0.72rem;
    font-weight: 700;
}
.citation-section {
    margin-top: 14px;
    padding-top: 12px;
    border-top: 1px solid var(--copa-border);
}
.citation-section .citation-label {
    font-size: 0.75rem;
    color: var(--copa-muted);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 8px;
}

/* ── Status chips in sidebar ─────────────────────────────────────────────── */
.doc-chip {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(16,163,127,0.15);
    border: 1px solid rgba(16,163,127,0.35);
    border-radius: 8px;
    padding: 6px 10px;
    margin-bottom: 5px;
    font-size: 0.78rem;
}

/* ── Input area ─────────────────────────────────────────────────────────── */
[data-testid="stChatInputContainer"] textarea {
    border-radius: 10px !important;
    border: 1.5px solid var(--copa-border) !important;
    font-size: 0.92rem !important;
}
[data-testid="stChatInputContainer"] textarea:focus {
    border-color: var(--copa-blue) !important;
    box-shadow: 0 0 0 2px rgba(0,48,135,0.12) !important;
}

/* ── Expander (sources) ─────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    font-size: 0.82rem !important;
    color: var(--copa-blue) !important;
    font-weight: 600 !important;
}

/* ── Hide Streamlit branding ─────────────────────────────────────────────── */
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
        st.error(
            f"**{label}** not found. "
            f"Add `{key}` in Streamlit Cloud → Settings → Secrets."
        )
        st.stop()
    return val


# ── Load RAG store (cached for app lifetime) ──────────────────────────────────
@st.cache_resource(show_spinner="✈️  Indexing knowledge base — first load takes ~30 seconds…")
def load_store():
    openai_key = _require_secret("OPENAI_API_KEY", "OpenAI API key")
    os.environ["OPENAI_API_KEY"] = openai_key

    from rag.vector_store import VectorStore
    from rag.document_loader import load_document
    from rag.chunker import chunk_records

    store   = VectorStore(persist_dir=None)   # in-memory, works on Streamlit Cloud
    indexed: list[str] = []

    repo_root = Path(__file__).parent
    for filename in REPO_DOCUMENTS:
        path = repo_root / filename
        if not path.exists():
            continue
        try:
            chunks = chunk_records(load_document(path))
            store.upsert(chunks)
            indexed.append(filename)
        except Exception as exc:
            st.warning(f"Could not index {filename}: {exc}")

    return store, indexed


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
    # Deduplicate by source+page
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
        f'<div class="citation-section">'
        f'<div class="citation-label">📎 Sources</div>'
        f'{pills}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo
    try:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/Copa_Airlines_logo.svg/320px-Copa_Airlines_logo.svg.png",
            use_container_width=True,
        )
    except Exception:
        st.markdown("### ✈️ Copa Airlines")
    st.markdown("&nbsp;")

    # Index status
    try:
        store, indexed_files = load_store()
        st.markdown("**Knowledge Base**")
        for f in indexed_files:
            st.markdown(
                f'<div class="doc-chip">✅ {f}</div>',
                unsafe_allow_html=True,
            )
        st.markdown(f"<small style='color:rgba(255,255,255,0.5)'>{store.count} chunks indexed</small>", unsafe_allow_html=True)
    except Exception as exc:
        st.error(f"Startup error: {exc}")

    st.divider()

    # Suggested questions
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
  <div style="margin-left:auto">
    <span class="copa-badge">RAG Powered</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages  = []
if "citations" not in st.session_state:
    st.session_state.citations = {}
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# ── Welcome card (only when no messages yet) ──────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
<div class="copa-welcome">
  <h3>👋 Welcome to the Copa Knowledge Assistant</h3>
  <p>
    Ask any question about Copa's systems — sign-in procedures, report types, cash accounting,
    transaction processing, and more. Every answer is drawn directly from official Copa documents
    and includes the exact page and section it came from.
    <br><br>
    Use the <strong>Suggested Questions</strong> in the sidebar to get started quickly,
    or type your own question below.
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
