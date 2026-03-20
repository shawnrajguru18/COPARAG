"""
Copa Airlines Sales Reporting - Knowledge Chatbot (Web UI)
Streamlit web interface powered by Claude and the Sales Reporting Solution Guide PDF.
"""

import os
import fitz  # pymupdf
import anthropic
import streamlit as st

PDF_PATH = os.path.join(os.path.dirname(__file__), "Sales Reporting Solution Guide_v1.2_2026MAR.pdf")

SYSTEM_PROMPT = """You are a knowledgeable assistant for Copa Airlines' Sales Reporting Manager system.
You answer questions based exclusively on the Sales Reporting Solution Guide provided to you.

Guidelines:
- Answer clearly and concisely using the information from the guide.
- If the answer is not found in the guide, say so honestly.
- When referencing steps or procedures, present them in a clear, numbered format.
- Do not fabricate information not present in the guide.
- Always be professional and helpful.
"""


@st.cache_resource
def load_knowledge_base() -> str:
    """Extract and cache PDF text."""
    doc = fitz.open(PDF_PATH)
    pages = []
    for page in doc:
        text = page.get_text().strip()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def get_system_prompt(knowledge_base: str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"## KNOWLEDGE BASE — Sales Reporting Solution Guide v1.2 (March 2026)\n\n"
        f"{knowledge_base}"
    )


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Copa Airlines | Sales Reporting Chatbot",
    page_icon="✈️",
    layout="centered",
)

st.title("✈️ Copa Airlines — Sales Reporting Chatbot")
st.caption("Ask anything about the Sales Reporting Manager system.")

# ── Load knowledge base ───────────────────────────────────────────────────────
knowledge_base = load_knowledge_base()
system_prompt = get_system_prompt(knowledge_base)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about the Sales Reporting Manager..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    api_key = st.secrets.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        st.error("ANTHROPIC_API_KEY not found. Add it in Streamlit Cloud → Settings → Secrets.")
        st.stop()
    client = anthropic.Anthropic(api_key=api_key)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        with client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=1024,
            system=system_prompt,
            messages=st.session_state.messages,
        ) as stream:
            for text in stream.text_stream:
                full_response += text
                response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
