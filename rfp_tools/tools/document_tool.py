"""
tools/document_tool.py
-----------------------
Document Tool: extracts clean text from an uploaded supplier RFP PDF.

Responsibility (per project brief):
    "Extracts clean text from each uploaded PDF." -> PyMuPDF / pypdf

This tool does NOT interpret or judge content — it only produces clean
text that the Evaluation Agent (LLM) can reason over.
"""

from __future__ import annotations
import io
import re

try:
    import fitz  # PyMuPDF
    _HAS_PYMUPDF = True
except ImportError:
    _HAS_PYMUPDF = False

try:
    from pypdf import PdfReader
    _HAS_PYPDF = True
except ImportError:
    _HAS_PYPDF = False


class DocumentExtractionError(Exception):
    """Raised when a PDF cannot be read by any available backend."""


def _extract_with_pymupdf(file_bytes: bytes) -> str:
    text_parts = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts)


def _extract_with_pypdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    text_parts = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(text_parts)


def _clean_text(raw_text: str) -> str:
    """Normalize whitespace and strip control characters so the LLM
    prompt is compact and consistent."""
    text = raw_text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_pdf(file_bytes: bytes, filename: str = "document.pdf") -> str:
    """
    Extract clean text from a PDF's raw bytes.

    Tries PyMuPDF first (better layout handling), falls back to pypdf.
    Raises DocumentExtractionError if neither backend is available or
    both fail, and if the resulting text is empty (e.g. a scanned PDF
    with no text layer — out of scope for this project, which uses
    synthetic text-based PDFs).
    """
    errors = []

    if _HAS_PYMUPDF:
        try:
            text = _extract_with_pymupdf(file_bytes)
            if text.strip():
                return _clean_text(text)
        except Exception as e:  # noqa: BLE001
            errors.append(f"PyMuPDF failed: {e}")

    if _HAS_PYPDF:
        try:
            text = _extract_with_pypdf(file_bytes)
            if text.strip():
                return _clean_text(text)
        except Exception as e:  # noqa: BLE001
            errors.append(f"pypdf failed: {e}")

    detail = "; ".join(errors) if errors else "No PDF backend installed."
    raise DocumentExtractionError(
        f"Could not extract text from '{filename}'. {detail}"
    )


def extract_text_from_uploaded_file(uploaded_file) -> str:
    """
    Convenience wrapper for Streamlit's UploadedFile objects
    (st.file_uploader). Reads bytes and delegates to
    extract_text_from_pdf.
    """
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)  # allow re-reading later in the app
    return extract_text_from_pdf(file_bytes, filename=uploaded_file.name)
