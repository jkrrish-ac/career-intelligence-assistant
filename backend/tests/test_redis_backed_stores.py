"""Parity tests: the Redis-backed rate limiter and conversation store must
behave identically to their in-process counterparts, since `app/api/deps.py`
picks one or the other based on `REDIS_URL` with no other code needing to
know which is running. Every test body below runs against *both*
implementations via parametrization, proving that parity rather than just
asserting it in a docstring.

Uses `fakeredis` (in-memory, no real Redis process needed) so these run
exactly like every other test here — no network, no external service.
"""

from __future__ import annotations

import fakeredis.aioredis
import pytest

from app.core.exceptions import RateLimitExceededError
from app.core.rate_limit import RedisRateLimiter, SlidingWindowRateLimiter
from app.services.conversation_store import ConversationStore, RedisConversationStore


def _fake_redis():
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


# --- Rate limiter parity -----------------------------------------------------

_RATE_LIMITER_IDS = ["in-memory", "redis (fakeredis)"]


def _rate_limiters(max_requests: int) -> list:
    return [
        lambda: SlidingWindowRateLimiter(max_requests=max_requests, window_seconds=60),
        lambda: RedisRateLimiter(redis_client=_fake_redis(), max_requests=max_requests, window_seconds=60),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("make_limiter", _rate_limiters(max_requests=3), ids=_RATE_LIMITER_IDS)
async def test_rate_limiter_allows_up_to_max_then_trips(make_limiter):
    limiter = make_limiter()

    for _ in range(3):
        await limiter.check("session-a")  # should not raise

    with pytest.raises(RateLimitExceededError):
        await limiter.check("session-a")


@pytest.mark.asyncio
@pytest.mark.parametrize("make_limiter", _rate_limiters(max_requests=1), ids=_RATE_LIMITER_IDS)
async def test_rate_limiter_keys_are_isolated(make_limiter):
    limiter = make_limiter()

    await limiter.check("session-a")  # uses up session-a's only slot
    await limiter.check("session-b")  # a different key must not be affected by that

    with pytest.raises(RateLimitExceededError):
        await limiter.check("session-a")


# --- Conversation store parity ------------------------------------------------

_STORE_IDS = ["in-memory", "redis (fakeredis)"]


def _conversation_stores(max_turns_per_session: int) -> list:
    return [
        lambda: ConversationStore(max_turns_per_session=max_turns_per_session),
        lambda: RedisConversationStore(
            redis_client=_fake_redis(),
            max_turns_per_session=max_turns_per_session,
            session_ttl_seconds=3600,
        ),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("make_store", _conversation_stores(max_turns_per_session=10), ids=_STORE_IDS)
async def test_conversation_store_preserves_order_and_isolation(make_store):
    store = make_store()

    await store.append("s1", "user", "What skills am I missing?")
    await store.append("s1", "assistant", "You're missing Kubernetes experience.")
    await store.append("s2", "user", "unrelated question")

    assert await store.get_history("s1") == [
        {"role": "user", "content": "What skills am I missing?"},
        {"role": "assistant", "content": "You're missing Kubernetes experience."},
    ]
    assert await store.get_history("s2") == [{"role": "user", "content": "unrelated question"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("make_store", _conversation_stores(max_turns_per_session=2), ids=_STORE_IDS)
async def test_conversation_store_caps_and_drops_oldest_first(make_store):
    store = make_store()

    for i in range(4):
        await store.append("s1", "user", f"question {i}")
        await store.append("s1", "assistant", f"answer {i}")

    history = await store.get_history("s1")
    assert len(history) == 4  # max_turns_per_session=2 -> 2 pairs -> 4 entries
    contents = [turn["content"] for turn in history]
    assert "question 0" not in contents, "oldest pair should have been evicted"
    assert "question 3" in contents


@pytest.mark.asyncio
@pytest.mark.parametrize("make_store", _conversation_stores(max_turns_per_session=10), ids=_STORE_IDS)
async def test_conversation_store_clear_removes_history(make_store):
    store = make_store()

    await store.append("s1", "user", "hi")
    await store.clear("s1")

    assert await store.get_history("s1") == []
