"""Session-scoped conversation history: an in-memory implementation and a
Redis-backed one behind the same interface (`get_history`/`append`/`clear`),
so `ChatService` doesn't care which is running underneath (see
`app/api/deps.py::get_conversation_store`, which picks based on whether
`REDIS_URL` is configured).

Both are async now (`ConversationStore`'s methods included, even though it
never awaits anything) so `ChatService` can call either one the same way —
see `core/rate_limit.py`'s module docstring for the same reasoning applied
there.
"""

from __future__ import annotations

import json
import threading
from collections import defaultdict, deque

from app.models.schemas import ConversationRole, ConversationTurn


class ConversationStore:
    """In-process only — resets on backend restart, isn't shared across
    processes. Acceptable for a single-instance take-home; see
    `RedisConversationStore` for the multi-instance/restart-durable
    equivalent."""

    def __init__(self, max_turns_per_session: int) -> None:
        # max_turns_per_session counts *turns* (one user + one assistant
        # message = 2 turns), so the deque cap is doubled here.
        self._max_len = max_turns_per_session * 2
        self._history: dict[str, deque[ConversationTurn]] = defaultdict(
            lambda: deque(maxlen=self._max_len)
        )
        self._lock = threading.Lock()

    async def get_history(self, session_id: str) -> list[ConversationTurn]:
        with self._lock:
            return list(self._history[session_id])

    async def append(self, session_id: str, role: ConversationRole, content: str) -> None:
        with self._lock:
            self._history[session_id].append({"role": role, "content": content})

    async def clear(self, session_id: str) -> None:
        with self._lock:
            self._history.pop(session_id, None)


class RedisConversationStore:
    """Same interface and eviction semantics as `ConversationStore` (capped
    at `max_turns_per_session * 2` entries), backed by a Redis `LIST` per
    session so history survives a restart and is shared across instances.

    `LPUSH` + `LTRIM` keep the list capped without a separate read-modify-
    write; a TTL refreshed on every write means an idle session's history
    eventually expires instead of accumulating forever."""

    def __init__(self, redis_client, max_turns_per_session: int, session_ttl_seconds: int) -> None:
        self._redis = redis_client
        self._max_len = max_turns_per_session * 2
        self._ttl_seconds = session_ttl_seconds

    @staticmethod
    def _key(session_id: str) -> str:
        return f"conversation:{session_id}"

    async def get_history(self, session_id: str) -> list[ConversationTurn]:
        # Stored newest-first (LPUSH), so reverse back to chronological order.
        raw = await self._redis.lrange(self._key(session_id), 0, -1)
        turns = [json.loads(item) for item in raw]
        turns.reverse()
        return turns

    async def append(self, session_id: str, role: ConversationRole, content: str) -> None:
        key = self._key(session_id)
        payload = json.dumps({"role": role, "content": content})
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.lpush(key, payload)
            pipe.ltrim(key, 0, self._max_len - 1)
            pipe.expire(key, self._ttl_seconds)
            await pipe.execute()

    async def clear(self, session_id: str) -> None:
        await self._redis.delete(self._key(session_id))
