import pytest

from app.services.conversation_store import ConversationStore


@pytest.mark.asyncio
async def test_history_is_empty_for_unseen_session():
    store = ConversationStore(max_turns_per_session=10)
    assert await store.get_history("new-session") == []


@pytest.mark.asyncio
async def test_append_preserves_order_within_a_session():
    store = ConversationStore(max_turns_per_session=10)
    await store.append("s1", "user", "What skills am I missing?")
    await store.append("s1", "assistant", "You're missing Kubernetes experience.")

    history = await store.get_history("s1")
    assert history == [
        {"role": "user", "content": "What skills am I missing?"},
        {"role": "assistant", "content": "You're missing Kubernetes experience."},
    ]


@pytest.mark.asyncio
async def test_sessions_are_isolated():
    store = ConversationStore(max_turns_per_session=10)
    await store.append("s1", "user", "hello from s1")
    await store.append("s2", "user", "hello from s2")

    assert await store.get_history("s1") == [{"role": "user", "content": "hello from s1"}]
    assert await store.get_history("s2") == [{"role": "user", "content": "hello from s2"}]


@pytest.mark.asyncio
async def test_history_is_capped_at_max_turns_and_drops_oldest_first():
    # max_turns_per_session=2 -> 2 user+assistant pairs -> 4 stored entries max
    store = ConversationStore(max_turns_per_session=2)
    for i in range(4):
        await store.append("s1", "user", f"question {i}")
        await store.append("s1", "assistant", f"answer {i}")

    history = await store.get_history("s1")
    assert len(history) == 4
    # The oldest pair (question 0 / answer 0) should have been evicted.
    contents = [turn["content"] for turn in history]
    assert "question 0" not in contents
    assert "question 3" in contents


@pytest.mark.asyncio
async def test_clear_removes_a_sessions_history():
    store = ConversationStore(max_turns_per_session=10)
    await store.append("s1", "user", "hi")
    await store.clear("s1")
    assert await store.get_history("s1") == []
