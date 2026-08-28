"""Sliding-window rate limiting: an in-process implementation and a
Redis-backed one behind the same `async def check(key)` interface, so
`ChatService` and every test fake using it don't need to know which is
running underneath (see `app/api/deps.py::get_rate_limiter`, which picks
based on whether `REDIS_URL` is configured).

`check()` is async on *both* implementations, even though the in-process one
never awaits anything — a uniform interface means callers never need an
`if redis: await ... else: ...` branch, and it costs nothing (an async
function with no `await` inside is not meaningfully slower to call than a
sync one).
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import defaultdict, deque

from app.core.exceptions import RateLimitExceededError


class SlidingWindowRateLimiter:
    """In-process only — enough for a single instance. Resets on restart and
    isn't shared across processes; see `RedisRateLimiter` below for the
    multi-instance/restart-durable equivalent."""

    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    async def check(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            window = self._hits[key]
            while window and now - window[0] > self._window_seconds:
                window.popleft()
            if len(window) >= self._max_requests:
                raise RateLimitExceededError(
                    f"Rate limit exceeded: max {self._max_requests} requests per "
                    f"{self._window_seconds}s.",
                    detail={"key": key},
                )
            window.append(now)


class RedisRateLimiter:
    """Same sliding-window semantics as `SlidingWindowRateLimiter`, but
    backed by a Redis sorted set so the count is shared across every backend
    instance and survives a restart -- what a multi-instance deployment
    actually needs (see PRD's cut list / parking lot).

    One sorted set per key, scored by request timestamp: `ZREMRANGEBYSCORE`
    evicts anything older than the window, `ZCARD` gets the current count.
    The count is checked *before* adding the current request's own entry, so
    a rejected request doesn't count against its own limit, matching the
    in-process version's semantics exactly (the deque there is only appended
    to *after* the length check passes)."""

    def __init__(self, redis_client, max_requests: int, window_seconds: int) -> None:
        self._redis = redis_client
        self._max_requests = max_requests
        self._window_seconds = window_seconds

    async def check(self, key: str) -> None:
        redis_key = f"ratelimit:{key}"
        now = time.time()
        window_start = now - self._window_seconds

        await self._redis.zremrangebyscore(redis_key, 0, window_start)
        count = await self._redis.zcard(redis_key)

        if count >= self._max_requests:
            raise RateLimitExceededError(
                f"Rate limit exceeded: max {self._max_requests} requests per "
                f"{self._window_seconds}s.",
                detail={"key": key},
            )

        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.zadd(redis_key, {str(uuid.uuid4()): now})
            # Belt-and-suspenders TTL so an abandoned key doesn't live
            # forever if this key is never hit again — the sliding-window
            # eviction above already keeps it small while active.
            pipe.expire(redis_key, self._window_seconds * 2)
            await pipe.execute()
