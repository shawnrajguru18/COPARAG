"""
Document loader — extracts plain text plus page/section metadata from PDF, DOCX, PPTX, TXT.

Each returned record looks like:
    {
        "source":   "filename.pdf",
        "page":     3,            # 1-based; for DOCX this is paragraph index
        "section":  "Section 2",  # best-effort heading extraction
        "text":     "raw text…"
    }
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


# ── PDF ────────────────────────────────────────────────────────────────────────

def _load_pdf(path: Path) -> list[dict[str, Any]]:
    import pymupdf  # fitz

    doc = pymupdf.open(str(path))
    records: list[dict[str, Any]] = []
    current_section = ""
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if not text:
            continue
        # Heuristic: first non-blank line on a page that is ALL-CAPS or short may be a heading
        first_line = text.splitlines()[0].strip()
        if first_line.isupper() or (len(first_line) < 80 and not first_line.endswith(".")):
            current_section = first_line
        records.append(
            {
                "source": path.name,
                "page": page_num,
                "section": current_section,
                "text": text,
            }
        )
    return records


# ── DOCX ───────────────────────────────────────────────────────────────────────

def _load_docx(path: Path) -> list[dict[str, Any]]:
    from docx import Document  # python-docx

    doc = Document(str(path))
    records: list[dict[str, Any]] = []
    current_section = ""
    buffer: list[str] = []

    def flush(section: str, idx: int) -> None:
        joined = " ".join(buffer).strip()
        if joined:
            records.append(
                {
                    "source": path.name,
                    "page": idx,
                    "section": section,
                    "text": joined,
                }
            )
        buffer.clear()

    for i, para in enumerate(doc.paragraphs):
        text = para.text.strip()
        if not text:
            continue
        style = para.style.name if para.style else ""
        if "Heading" in style:
            flush(current_section, len(records) + 1)
            current_section = text
        else:
            buffer.append(text)

    flush(current_section, len(records) + 1)
    return records


# ── PPTX ───────────────────────────────────────────────────────────────────────

def _load_pptx(path: Path) -> list[dict[str, Any]]:
    from pptx import Presentation

    prs = Presentation(str(path))
    records: list[dict[str, Any]] = []
    for slide_num, slide in enumerate(prs.slides, start=1):
        texts: list[str] = []
        title = ""
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            shape_text = "\n".join(
                p.text for p in shape.text_frame.paragraphs if p.text.strip()
            )
            if shape.shape_type == 13 or (shape.name and "title" in shape.name.lower()):
                title = shape_text
            else:
                texts.append(shape_text)
        combined = "\n".join(t for t in texts if t)
        if combined:
            records.append(
                {
                    "source": path.name,
                    "page": slide_num,
                    "section": title,
                    "text": combined,
                }
            )
    return records


# ── TXT ────────────────────────────────────────────────────────────────────────

def _load_txt(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(errors="replace")
    # Split on double newlines to get logical paragraphs
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    return [
        {"source": path.name, "page": i + 1, "section": "", "text": para}
        for i, para in enumerate(paragraphs)
    ]


# ── Dispatcher ────────────────────────────────────────────────────────────────

def load_document(path: str | Path) -> list[dict[str, Any]]:
    """
    Load *path* and return a list of page/paragraph records with metadata.
    Raises ``ValueError`` for unsupported file types.
    """
    path = Path(path)
    ext = path.suffix.lower()
    loaders = {
        ".pdf": _load_pdf,
        ".docx": _load_docx,
        ".doc": _load_docx,   # requires anti-word or LibreOffice pre-conversion for .doc
        ".pptx": _load_pptx,
        ".txt": _load_txt,
    }
    loader = loaders.get(ext)
    if loader is None:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader(path)
