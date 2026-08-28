"""In-process sliding-window rate limiter.

Enough to demonstrate the guardrail exists for a single-instance take-home;
a real deployment would swap this for a Redis-backed limiter shared across
processes (named in the PRD's cut list, not silently missing).
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from app.core.exceptions import RateLimitExceededError


class SlidingWindowRateLimiter:
    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
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
