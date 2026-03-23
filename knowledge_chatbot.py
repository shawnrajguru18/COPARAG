"""
Copa Airlines Sales Reporting - Knowledge Chatbot
Answers questions based on the Sales Reporting Solution Guide PDF.
"""

import os
import fitz  # pymupdf
import anthropic

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


def extract_pdf_text(pdf_path: str) -> str:
    """Extract all text from the PDF document."""
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        text = page.get_text().strip()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def run_chatbot():
    """Run the knowledge chatbot in an interactive loop."""
    print("Loading Sales Reporting Solution Guide...")
    knowledge_base = extract_pdf_text(PDF_PATH)
    print(f"Loaded {len(knowledge_base):,} characters from the guide.\n")

    client = anthropic.Anthropic()
    messages = []

    system_with_knowledge = (
        f"{SYSTEM_PROMPT}\n\n"
        f"## KNOWLEDGE BASE — Sales Reporting Solution Guide v1.2 (March 2026)\n\n"
        f"{knowledge_base}"
    )

    print("=" * 60)
    print("  Copa Airlines — Sales Reporting Knowledge Chatbot")
    print("=" * 60)
    print("Ask any question about the Sales Reporting Manager.")
    print("Type 'exit' or 'quit' to end the session.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAssistant: Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "bye", "goodbye"):
            print("\nAssistant: Goodbye! Feel free to return if you have more questions.")
            break

        messages.append({"role": "user", "content": user_input})

        print("\nAssistant: ", end="", flush=True)

        with client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=1024,
            system=system_with_knowledge,
            messages=messages,
        ) as stream:
            full_response = ""
            for text in stream.text_stream:
                print(text, end="", flush=True)
                full_response += text

        print("\n")
        messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    run_chatbot()
