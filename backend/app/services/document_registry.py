"""Lightweight persistent registry of uploaded document metadata.

Chroma stores chunks; it isn't a great place to ask "what documents exist"
without loading vectors, so document-level metadata (label, upload time,
chunk count) lives in its own small JSON-backed store. Simple by design —
a take-home doesn't need a second database for a handful of documents, and
this is the kind of scope call worth naming rather than justifying at length.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from app.core.logging import get_logger
from app.models.schemas import DocumentMetadata

logger = get_logger(__name__)


class DocumentRegistry:
    def __init__(self, storage_path: Path) -> None:
        self._storage_path = storage_path
        self._lock = threading.Lock()
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._documents: dict[str, DocumentMetadata] = self._load()

    def _load(self) -> dict[str, DocumentMetadata]:
        if not self._storage_path.exists():
            return {}
        raw = json.loads(self._storage_path.read_text())
        return {doc_id: DocumentMetadata(**data) for doc_id, data in raw.items()}

    def _persist(self) -> None:
        payload = {doc_id: json.loads(doc.model_dump_json()) for doc_id, doc in self._documents.items()}
        self._storage_path.write_text(json.dumps(payload, indent=2))

    def add(self, document: DocumentMetadata) -> None:
        with self._lock:
            self._documents[document.document_id] = document
            self._persist()
        logger.info("document_registered", document_id=document.document_id, label=document.label)

    def list_all(self) -> list[DocumentMetadata]:
        with self._lock:
            return list(self._documents.values())

    def get(self, document_id: str) -> DocumentMetadata | None:
        with self._lock:
            return self._documents.get(document_id)

    def delete(self, document_id: str) -> None:
        with self._lock:
            self._documents.pop(document_id, None)
            self._persist()
