"""
Copa Airlines Sales Reporting - Knowledge Chatbot (Web UI)
Streamlit web interface powered by Claude and the Sales Reporting Solution Guide PDF.
"""

import os
import re
import fitz  # pymupdf
import anthropic
import streamlit as st

PDF_PATH = os.path.join(os.path.dirname(__file__), "Sales Reporting Solution Guide_v1.2_2026MAR.pdf")

SYSTEM_PROMPT = """You are a precise, expert assistant for Copa Airlines' Sales Reporting Manager system.
You answer questions based exclusively on the Sales Reporting Solution Guide provided.

## RESPONSE FORMAT RULES — follow these strictly for every answer:

1. **Always open with a one-sentence direct answer** to the question.

2. **Use structured sections** with bold headers (e.g., **Steps to Follow**, **Key Requirements**, **Important Notes**).

3. **For any procedure or workflow, always use a numbered step list**, e.g.:
   **Steps:**
   1. Go to the Sales Reporting module from the main DXC CSS GUI.
   2. Enter your credentials.
   3. …

4. **For comparisons or multiple items, always use a markdown table**, e.g.:
   | Feature | Detail |
   |---------|--------|
   | ... | ... |

5. **Use callout blocks for warnings, tips, and notes**, formatted exactly as:
   > ⚠️ **Warning:** …
   > 💡 **Tip:** …
   > ℹ️ **Note:** …

6. **Be specific** — always include exact menu names, button labels, field names, and screen names as they appear in the guide. Never give vague or generic answers.

7. If a process has sub-steps, use nested lists.

8. End every answer with a **> 💡 Tip:** callout giving a practical reminder or the most common mistake to avoid.

9. If the answer is not in the guide, say exactly: "This topic is not covered in the Sales Reporting Solution Guide v1.2. Please contact your system administrator."
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


def render_response(text: str):
    """
    Render a Claude response with enriched Streamlit components.
    Detects callout blocks (> ⚠️, > 💡, > ℹ️) and renders them
    as st.warning / st.success / st.info. Everything else is markdown.
    """
    callout_pattern = re.compile(
        r'^((?:> .+\n?)+)',
        re.MULTILINE,
    )

    parts = callout_pattern.split(text)
    for part in parts:
        if not part.strip():
            continue
        if part.startswith(">"):
            # Strip the "> " prefix from each line
            lines = [line.lstrip("> ").rstrip() for line in part.strip().splitlines()]
            block_text = "\n".join(lines)
            first = block_text.lstrip()
            if first.startswith("⚠️"):
                st.warning(block_text)
            elif first.startswith("💡"):
                st.success(block_text)
            elif first.startswith("ℹ️"):
                st.info(block_text)
            else:
                st.info(block_text)
        else:
            st.markdown(part)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Copa Airlines | Sales Reporting Chatbot",
    page_icon="✈️",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/Copa_Airlines_logo.svg/320px-Copa_Airlines_logo.svg.png",
        use_container_width=True,
    )
    st.markdown("## Sales Reporting Guide")
    st.caption("v1.2 — March 2026")
    st.divider()

    st.markdown("### Suggested Questions")

    topics = {
        "Getting Started": [
            "How do I sign in to the Sales Reporting Manager?",
            "What are the hardware and browser requirements?",
            "How do I sign out?",
        ],
        "Reports": [
            "How do I close an Agent Sales Report?",
            "What transactions appear in the Agent Sales Report?",
            "How do I view a Station Sales Report?",
            "What is the difference between an Agent and a Supervisor report?",
        ],
        "Filters & Search": [
            "How do I filter reports by agent?",
            "How do I search for a specific transaction?",
            "How do I view multiple station reports?",
        ],
        "Cash Accounting": [
            "How does the Cash Accounting Report work?",
            "What is a variance in a Cash Accounting Report?",
            "How do I balance the Cash Accounting Report?",
        ],
        "Transactions": [
            "What is the difference between a void and a refund in the Sales Report?",
            "How are EMDs recorded in the Sales Report?",
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
st.title("✈️ Copa Airlines — Sales Reporting Chatbot")
st.caption("Powered by the Sales Reporting Solution Guide v1.2 · Ask anything about the system.")

# ── Load knowledge base ───────────────────────────────────────────────────────
knowledge_base = load_knowledge_base()
system_prompt = get_system_prompt(knowledge_base)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_response(msg["content"])
        else:
            st.markdown(msg["content"])

# ── Resolve a sidebar button click as a prompt ────────────────────────────────
active_prompt = None
if st.session_state.pending_prompt:
    active_prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

# ── Chat input ────────────────────────────────────────────────────────────────
typed_prompt = st.chat_input("Ask a question about the Sales Reporting Manager...")
prompt = typed_prompt or active_prompt

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    api_key = st.secrets.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        st.error("ANTHROPIC_API_KEY not found. Add it in Streamlit Cloud → Settings → Secrets.")
        st.stop()
    client = anthropic.Anthropic(api_key=api_key)

    with st.chat_message("assistant"):
        stream_placeholder = st.empty()
        full_response = ""

        with client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=2048,
            system=system_prompt,
            messages=st.session_state.messages,
        ) as stream:
            for text in stream.text_stream:
                full_response += text
                stream_placeholder.markdown(full_response + "▌")

        # Replace the raw stream with the rich rendered version
        stream_placeholder.empty()
        render_response(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
