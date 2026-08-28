"""Application configuration.

Every tunable that plausibly needs to change between a laptop, a reviewer's
machine, and a future deployment lives here, backed by environment variables
with sensible defaults. Nothing here should require a code change to retune.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ---
    anthropic_api_key: str = Field(default="", description="Anthropic API key for Claude calls")
    claude_model: str = Field(default="claude-sonnet-4-6")
    claude_max_tokens: int = Field(default=1024)

    # --- Embeddings ---
    embedding_provider: str = Field(default="local", description="local | voyage | openai")
    embedding_model: str = Field(default="all-MiniLM-L6-v2")

    # --- Reranking ---
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    rerank_enabled: bool = Field(default=True)

    # --- Vector store ---
    chroma_persist_dir: Path = Field(default=Path("./data/chroma"))
    chroma_collection_name: str = Field(default="career_intelligence")

    # --- Chunking ---
    chunk_size_tokens: int = Field(default=500)
    chunk_overlap_tokens: int = Field(default=50)

    # --- Retrieval ---
    retrieval_candidate_k: int = Field(default=10, description="Candidates pulled before fusion/rerank")
    retrieval_final_k: int = Field(default=5, description="Chunks kept after rerank, sent to the LLM")

    # --- Uploads ---
    max_file_size_mb: int = Field(default=10)
    allowed_upload_extensions: tuple[str, ...] = Field(default=(".pdf", ".docx", ".txt"))
    upload_dir: Path = Field(default=Path("./data/uploads"))

    # --- Rate limiting ---
    rate_limit_requests: int = Field(default=20, description="Max /chat requests per window")
    rate_limit_window_seconds: int = Field(default=60)

    # --- Conversation memory ---
    max_history_turns: int = Field(
        default=10, description="User+assistant turn pairs kept per session, in-memory only"
    )

    # --- Server ---
    cors_allow_origins: tuple[str, ...] = Field(default=("http://localhost:5173",))
    log_level: str = Field(default="INFO")


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton — read once, reused everywhere via DI."""
    return Settings()
