"""In-memory, session-scoped conversation history.

Deliberately not persisted to disk (unlike documents/vectors) — conversation
memory resets on backend restart, which is an acceptable tradeoff for a
take-home and is called out here rather than silently assumed. A real
deployment would move this to Redis alongside a real rate limiter (see
app/core/rate_limit.py's docstring for the same tradeoff).
"""

from __future__ import annotations

import threading
from collections import defaultdict, deque

from app.models.schemas import ConversationRole, ConversationTurn


class ConversationStore:
    def __init__(self, max_turns_per_session: int) -> None:
        # max_turns_per_session counts *turns* (one user + one assistant
        # message = 2 turns), so the deque cap is doubled here.
        self._max_len = max_turns_per_session * 2
        self._history: dict[str, deque[ConversationTurn]] = defaultdict(
            lambda: deque(maxlen=self._max_len)
        )
        self._lock = threading.Lock()

    def get_history(self, session_id: str) -> list[ConversationTurn]:
        with self._lock:
            return list(self._history[session_id])

    def append(self, session_id: str, role: ConversationRole, content: str) -> None:
        with self._lock:
            self._history[session_id].append({"role": role, "content": content})

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._history.pop(session_id, None)
