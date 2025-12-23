"""
PDF data processing utilities.

Responsibilities:
- Load PDF reports from `data/raw_documents`
- Extract text page-by-page (to preserve page citations)
- Split into chunks for retrieval, preserving page metadata
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import hashlib
import PyPDF2

from .config import RAW_DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class DocumentChunk:
    """Lightweight container for a chunk of text and its source metadata."""
    text: str
    source: str                 # filename only (better for citations)
    page: int                   # 1-indexed page number
    chunk_id: str               # stable id for upserts


def list_pdf_files(directory: Path | None = None) -> List[Path]:
    """Return a list of PDF files under the given directory (non-recursive)."""
    directory = directory or RAW_DOCS_DIR
    return sorted(p for p in directory.glob("*.pdf") if p.is_file())


def simple_text_splitter(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """Simple character-based splitter with overlap."""
    text = (text or "").strip()
    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - chunk_overlap

    return chunks


def extract_pages_from_pdf(pdf_path: Path) -> List[str]:
    """
    Extract text from a PDF page-by-page.

    Returns a list of page texts (index 0 corresponds to page 1).
    """
    pages: List[str] = []
    with pdf_path.open("rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            pages.append(page.extract_text() or "")
    return pages


def generate_chunks_from_pdfs(
    pdf_paths: Iterable[Path] | None = None,
) -> List[DocumentChunk]:
    """
    End-to-end helper:
    - iterate PDFs
    - extract text page-by-page
    - split each page into chunks
    - return DocumentChunk objects with correct page numbers
    """
    pdf_paths = list(pdf_paths or list_pdf_files())
    chunks: List[DocumentChunk] = []

    for pdf_path in pdf_paths:
        source_name = pdf_path.name  # keep citations clean
        pages = extract_pages_from_pdf(pdf_path)

        for page_number, page_text in enumerate(pages, start=1):
            raw_chunks = simple_text_splitter(page_text)

            for idx, c in enumerate(raw_chunks):
                # stable chunk id (good for Chroma upsert)
                stable = f"{source_name}|p{page_number}|c{idx}"
                chunk_hash = hashlib.md5(stable.encode("utf-8")).hexdigest()

                chunks.append(
                    DocumentChunk(
                        text=c,
                        source=source_name,
                        page=page_number,
                        chunk_id=chunk_hash,
                    )
                )

    return chunks




