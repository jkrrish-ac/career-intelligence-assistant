"""Document text extraction.

One function per format behind a common `parse(path) -> str` interface so the
ingestion service doesn't need to know or care what format it received.
"""

from __future__ import annotations

from pathlib import Path

import docx
from pypdf import PdfReader

from app.core.exceptions import DocumentParseError, UnsupportedFileTypeError
from app.core.logging import get_logger

logger = get_logger(__name__)


def parse_pdf(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(pages).strip()
    except Exception as exc:  # pypdf raises a variety of format-specific errors
        raise DocumentParseError(
            f"Could not extract text from PDF '{path.name}'", detail={"filename": path.name}
        ) from exc

    if not text:
        raise DocumentParseError(
            f"PDF '{path.name}' produced no extractable text "
            "(likely a scanned image without a text layer — OCR is out of scope)",
            detail={"filename": path.name},
        )
    return text


def parse_docx(path: Path) -> str:
    try:
        document = docx.Document(str(path))
        paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
        text = "\n".join(paragraphs).strip()
    except Exception as exc:
        raise DocumentParseError(
            f"Could not extract text from DOCX '{path.name}'", detail={"filename": path.name}
        ) from exc

    if not text:
        raise DocumentParseError(
            f"DOCX '{path.name}' produced no extractable text", detail={"filename": path.name}
        )
    return text


def parse_txt(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception as exc:
        raise DocumentParseError(
            f"Could not read text file '{path.name}'", detail={"filename": path.name}
        ) from exc

    if not text:
        raise DocumentParseError(f"'{path.name}' is empty", detail={"filename": path.name})
    return text


_PARSERS = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".txt": parse_txt,
}


def parse_document(path: Path) -> str:
    """Dispatch to the right parser based on file extension."""
    suffix = path.suffix.lower()
    parser = _PARSERS.get(suffix)
    if parser is None:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{suffix}'. Allowed: {', '.join(_PARSERS)}",
            detail={"filename": path.name, "suffix": suffix},
        )
    logger.info("parsing_document", filename=path.name, suffix=suffix)
    return parser(path)
