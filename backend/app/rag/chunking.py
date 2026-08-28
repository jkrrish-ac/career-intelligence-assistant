"""Chunking strategy: section-aware first, recursive-split fallback.

Approach (per PRD §5):
1. Try to split on structural cues — common resume/JD section headers and
   blank-line-separated bullet groups.
2. Within each detected section, if it's still bigger than `chunk_size`,
   recursively split on paragraph -> line -> word boundaries with overlap.
3. If no headers are detected at all (unstructured document), skip straight
   to recursive splitting over the whole text.

Note on units: `chunk_size`/`chunk_overlap` are expressed in whitespace-
separated words as a token proxy. This avoids pulling in a tokenizer just for
chunking; it's a documented approximation, not hidden behavior — a real
tokenizer count is a one-line swap in `_word_count` if precision matters more
than speed later.
"""

from __future__ import annotations

import re
import uuid

from app.core.logging import get_logger
from app.models.schemas import Chunk, SourceType

logger = get_logger(__name__)

# Common resume/JD section headers, matched case-insensitively against a
# line on its own (optionally followed by a colon).
_SECTION_HEADER_PATTERN = re.compile(
    r"^\s*(SUMMARY|OBJECTIVE|EXPERIENCE|WORK EXPERIENCE|PROFESSIONAL EXPERIENCE|"
    r"EMPLOYMENT HISTORY|EDUCATION|SKILLS|TECHNICAL SKILLS|PROJECTS|CERTIFICATIONS|"
    r"REQUIREMENTS|QUALIFICATIONS|RESPONSIBILITIES|ABOUT (?:THE ROLE|US)|"
    r"WHAT YOU('?LL| WILL) DO|WHO YOU ARE|BENEFITS|NICE TO HAVE[S]?)\s*:?\s*$",
    re.IGNORECASE,
)


def _split_into_sections(text: str) -> list[tuple[str | None, str]]:
    """Return [(section_name, section_text), ...]. section_name is None for
    any leading text before the first recognized header."""
    lines = text.splitlines()
    sections: list[tuple[str | None, list[str]]] = []
    current_name: str | None = None
    current_lines: list[str] = []

    for line in lines:
        match = _SECTION_HEADER_PATTERN.match(line)
        if match:
            if current_lines:
                sections.append((current_name, current_lines))
            current_name = match.group(1).strip().title()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_name, current_lines))

    return [(name, "\n".join(body).strip()) for name, body in sections if "\n".join(body).strip()]


def _word_count(text: str) -> int:
    return len(text.split())


def _recursive_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split on paragraph -> line -> word boundaries, respecting chunk_size
    (in words) with chunk_overlap carried between consecutive chunks."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text] if text.strip() else []

    chunks: list[str] = []
    start = 0
    step = max(chunk_size - chunk_overlap, 1)
    while start < len(words):
        window = words[start : start + chunk_size]
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks


def chunk_document(
    *,
    text: str,
    document_id: str,
    source_type: SourceType,
    label: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Turn raw extracted text into metadata-tagged Chunks."""
    sections = _split_into_sections(text)

    if not sections:
        logger.info("chunking_fallback_no_headers", document_id=document_id, label=label)
        sections = [(None, text)]
    else:
        logger.info(
            "chunking_sections_detected",
            document_id=document_id,
            label=label,
            sections=[name for name, _ in sections],
        )

    chunks: list[Chunk] = []
    for section_name, section_text in sections:
        pieces = (
            [section_text]
            if _word_count(section_text) <= chunk_size
            else _recursive_split(section_text, chunk_size, chunk_overlap)
        )
        for piece in pieces:
            if not piece.strip():
                continue
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document_id,
                    source_type=source_type,
                    label=label,
                    section=section_name,
                    text=piece.strip(),
                )
            )

    logger.info("chunking_complete", document_id=document_id, label=label, chunk_count=len(chunks))
    return chunks
