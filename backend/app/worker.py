"""arq worker for out-of-band document ingestion.

Used only when `INGESTION_MODE=async` (see `app/api/documents.py`'s POST
route and `app/services/ingestion_service.py`'s `register_pending`/
`process_pending`): the route registers a `pending` document and enqueues a
job here instead of doing the parse/chunk/embed work inline, so a slow PDF
or a burst of uploads doesn't block the HTTP response.

Run with: `arq app.worker.WorkerSettings` (docker-compose's `worker` service
does exactly this, sharing the backend image -- see docker-compose.yml).
"""

from __future__ import annotations

from arq.connections import RedisSettings

from app.api.deps import get_document_registry, get_embedding_provider_dep, get_vector_store
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.services.ingestion_service import IngestionService

logger = get_logger(__name__)


async def run_ingestion(ctx: dict, document_id: str) -> None:
    """The arq task itself. Reuses the exact same
    `IngestionService.process_pending()` pipeline the app would use for any
    other ingestion path -- this function's only job is to run it out of
    the request path, not to reimplement parsing/chunking/embedding."""
    ingestion_service: IngestionService = ctx["ingestion_service"]
    await ingestion_service.process_pending(document_id)


async def on_startup(ctx: dict) -> None:
    """Builds the same IngestionService the FastAPI app builds (via the
    same `app.api.deps` factory functions, so there's exactly one place
    that knows how to construct these collaborators), once per worker
    process rather than once per job."""
    settings = get_settings()
    configure_logging(settings.log_level)
    ctx["ingestion_service"] = IngestionService(
        embedding_provider=get_embedding_provider_dep(),
        vector_store=get_vector_store(),
        document_registry=get_document_registry(),
        upload_dir=settings.upload_dir,
        chunk_size=settings.chunk_size_tokens,
        chunk_overlap=settings.chunk_overlap_tokens,
        max_file_size_mb=settings.max_file_size_mb,
    )
    logger.info("worker_startup")


async def on_shutdown(ctx: dict) -> None:
    logger.info("worker_shutdown")


_settings = get_settings()
# Falls back to RedisSettings()'s localhost default when REDIS_URL isn't
# set, purely so `import app.worker` doesn't blow up in a context where
# Redis isn't configured (e.g. a test importing this module) -- actually
# *running* this worker without REDIS_URL set is a misconfiguration the
# worker process will surface immediately as a connection error, which is
# the right failure mode (loud and at startup, not silent).
_redis_settings = (
    RedisSettings.from_dsn(_settings.redis_url) if _settings.redis_url else RedisSettings()
)


class WorkerSettings:
    functions = [run_ingestion]
    on_startup = on_startup
    on_shutdown = on_shutdown
    redis_settings = _redis_settings
