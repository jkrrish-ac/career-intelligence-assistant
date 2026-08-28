"""Embedding provider abstraction.

`EmbeddingProvider` is the seam the PRD calls out explicitly: the default
implementation is local and free (sentence-transformers), but swapping in a
hosted provider (Voyage, OpenAI) later is a new class + a config change, not
a rewrite of the ingestion or retrieval code that calls it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache

from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingProvider(ABC):
    """Interface every embedding backend must satisfy."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text, same order."""

    @property
    @abstractmethod
    def dimension(self) -> int: ...


class LocalEmbeddingProvider(EmbeddingProvider):
    """sentence-transformers running on CPU — no API key, no network at
    request time (the model is downloaded once, ideally baked into the
    Docker image at build time so `docker-compose up` doesn't stall)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        logger.info("loading_embedding_model", model_name=model_name)
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return vectors.tolist()

    @property
    def dimension(self) -> int:
        return self._dimension


@lru_cache
def get_embedding_provider(provider: str = "local", model_name: str = "all-MiniLM-L6-v2") -> EmbeddingProvider:
    """Factory + cache — the model load is expensive, do it once per process."""
    if provider == "local":
        return LocalEmbeddingProvider(model_name=model_name)
    raise NotImplementedError(
        f"Embedding provider '{provider}' is not implemented yet. "
        "The EmbeddingProvider interface supports adding it — see app/rag/embeddings.py."
    )
