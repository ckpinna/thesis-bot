from __future__ import annotations

from pathlib import Path

import fitz


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    try:
        text = ""
        for page_num in range(len(doc)):
            page = doc[page_num]
            text += page.get_text()
        return text
    finally:
        doc.close()


def load_pdf_texts(data_folder: Path) -> dict[str, str]:
    """Load all PDFs from a folder and return a filename -> text mapping."""
    pdf_texts: dict[str, str] = {}
    pdf_files = sorted(data_folder.glob("*.pdf"))

    if not pdf_files:
        print(f"WARNING: No PDF files found in {data_folder}")
        return pdf_texts

    for pdf_file in pdf_files:
        print(f"Extracting text from: {pdf_file.name}")
        try:
            text = extract_text_from_pdf(pdf_file)
            pdf_texts[pdf_file.name] = text
            print(f"  Loaded {len(text):,} characters")
        except Exception as exc:
            print(f"  Failed to extract {pdf_file.name}: {exc}")

    return pdf_texts

