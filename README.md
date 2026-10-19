# Copa Airlines — Knowledge Assistant (RAG)

A Retrieval-Augmented Generation chatbot that answers questions by searching indexed documents and citing exact sources. Built on ChromaDB, OpenAI embeddings, and Claude.

---

## Architecture

```
SharePoint / local PDFs
        │
        ▼
  rag/document_loader.py   ← extract text + page metadata
        │
        ▼
  rag/chunker.py           ← split into ~400-token chunks with overlap
        │
        ▼
  rag/vector_store.py      ← embed (text-embedding-3-small) + store (ChromaDB)
        │
  ┌─────┴──────┐
  │  Query time│
  └─────┬──────┘
        │  user question
        ▼
  rag/vector_store.py      ← embed question → top-6 similar chunks
        │
        ▼
  rag/chain.py             ← build context block with [DOC-N] labels → Claude
        │
        ▼
  app.py (Streamlit)       ← stream response + render citations table
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
cp .env.example .env
# Edit .env and fill in ANTHROPIC_API_KEY and OPENAI_API_KEY
```

### 3. Ingest documents

**Local files:**
```bash
python ingest.py --local "Sales Reporting Solution Guide_v1.2_2026MAR.pdf"
```

**From SharePoint** (requires Azure AD credentials in `.env`):
```bash
python ingest.py --sharepoint
```

### 4. Run the app

```bash
streamlit run app.py
```

---

## SharePoint Setup

The SharePoint ingestion uses the **Microsoft Graph API** with an Azure AD App Registration (client credentials flow — no user login required).

1. **Create an App Registration** in Azure AD:
   - API Permissions → Microsoft Graph → Application permissions:
     - `Sites.Read.All`
     - `Files.Read.All`
   - Grant admin consent.

2. **Find your SharePoint Site ID:**
   ```bash
   python -m rag.sharepoint_client --list-sites
   ```

3. **Set environment variables** (see `.env.example`).

4. **Run ingestion:**
   ```bash
   python ingest.py --sharepoint
   ```

Re-run ingestion whenever documents change. The pipeline is incremental: it deletes stale chunks for each re-ingested file before adding new ones.

---

## How Citations Work

Each document chunk is stored with its source filename, page number, and section heading. At query time:

1. The top-6 most relevant chunks are retrieved.
2. Each chunk is labelled `[DOC-1]` … `[DOC-N]` in the context block sent to Claude.
3. Claude is instructed to cite labels inline and append a `## Sources` section.
4. The Streamlit UI renders a **Sources used** expander below each answer showing document name, page, and section.

---

## Supported Document Types

| Format | Handler |
|--------|---------|
| `.pdf` | PyMuPDF (page-by-page extraction) |
| `.docx` | python-docx (paragraph/heading-aware) |
| `.pptx` | python-pptx (slide-by-slide) |
| `.txt`  | built-in (paragraph splitting) |

---

## Re-indexing

To re-index a single file (e.g. after an update):
```bash
python ingest.py --local updated_guide.pdf
```
The pipeline removes all old chunks for that filename before indexing the new version.

To wipe the entire store and start fresh:
```bash
rm -rf ./chroma_db
python ingest.py --local ...
```
