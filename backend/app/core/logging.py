"""Structured logging setup.

Every log line is JSON with consistent keys, not a print statement, so a
reviewer (or a future on-call engineer) can grep/filter by field instead of
parsing prose. `timed` wraps any function with start/end/duration logging,
used on the three spans that matter most for RAG observability: retrieval,
rerank, and the LLM call.
"""

import functools
import logging
import sys
import time
from collections.abc import Callable
from typing import Any, TypeVar

import structlog

F = TypeVar("F", bound=Callable[..., Any])


def configure_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper(), logging.INFO),
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


_timing_log = get_logger("timing")


def timed(span_name: str) -> Callable[[F], F]:
    """Decorator that logs {span, duration_ms, ok} for the wrapped call.

    Applied to retrieval, rerank, and LLM-call functions so every request's
    cost breakdown is visible in the logs without instrumenting each call
    site by hand.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            ok = True
            try:
                return await func(*args, **kwargs)
            except Exception:
                ok = False
                raise
            finally:
                duration_ms = round((time.perf_counter() - start) * 1000, 2)
                _timing_log.info("span_complete", span=span_name, duration_ms=duration_ms, ok=ok)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            ok = True
            try:
                return func(*args, **kwargs)
            except Exception:
                ok = False
                raise
            finally:
                duration_ms = round((time.perf_counter() - start) * 1000, 2)
                _timing_log.info("span_complete", span=span_name, duration_ms=duration_ms, ok=ok)

        if _is_coroutine_function(func):
            return async_wrapper  # type: ignore[return-value]
        return sync_wrapper  # type: ignore[return-value]

    return decorator


def _is_coroutine_function(func: Callable[..., Any]) -> bool:
    import asyncio

    return asyncio.iscoroutinefunction(func)
