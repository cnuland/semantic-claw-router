"""Tests for request deduplication."""

import time

import pytest

from semantic_claw_router.pipeline.dedup import (
    DedupCache,
    DedupStatus,
    canonicalize_request,
    hash_request,
)


class TestCanonicalization:
    def test_strips_volatile_fields(self):
        body1 = {
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}],
            "timestamp": "2024-01-01T00:00:00Z",
            "request_id": "abc123",
        }
        body2 = {
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}],
            "timestamp": "2024-01-02T00:00:00Z",
            "request_id": "xyz789",
        }
        assert canonicalize_request(body1) == canonicalize_request(body2)

    def test_sorts_keys(self):
        body1 = {"model": "test", "messages": [], "stream": False}
        body2 = {"stream": False, "messages": [], "model": "test"}
        assert canonicalize_request(body1) == canonicalize_request(body2)

    def test_different_content_different_hash(self):
        body1 = {"messages": [{"role": "user", "content": "hello"}]}
        body2 = {"messages": [{"role": "user", "content": "world"}]}
        h1 = hash_request(canonicalize_request(body1))
        h2 = hash_request(canonicalize_request(body2))
        assert h1 != h2

    def test_nested_volatile_stripped(self):
        body = {
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {"request_id": "abc", "other": "keep"},
        }
        canonical = canonicalize_request(body)
        assert "abc" not in canonical
        assert "keep" in canonical


class TestDedupCache:
    def test_first_request_returns_none(self):
        cache = DedupCache()
        body = {"messages": [{"role": "user", "content": "hello"}]}
        h, entry = cache.check(body)
        assert entry is None
        assert h  # Hash should be non-empty

    def test_pending_then_completed(self):
        cache = DedupCache()
        body = {"messages": [{"role": "user", "content": "hello"}]}

        # First check → pending
        h, entry = cache.check(body)
        assert entry is None

        # Complete it
        response = {"choices": [{"message": {"content": "hi"}}]}
        cache.complete(h, response)

        # Second check → completed with cached response
        h2, entry2 = cache.check(body)
        assert entry2 is not None
        assert entry2.status == DedupStatus.COMPLETED
        assert entry2.response == response

    def test_duplicate_returns_cached(self):
        cache = DedupCache()
        body = {"messages": [{"role": "user", "content": "test"}]}

        h, _ = cache.check(body)
        cache.complete(h, {"result": "cached"})

        # Retry with same content but different metadata
        body_retry = {
            "messages": [{"role": "user", "content": "test"}],
            "timestamp": "different",
        }
        _, entry = cache.check(body_retry)
        assert entry is not None
        assert entry.response == {"result": "cached"}

    def test_ttl_expiration(self):
        cache = DedupCache(window_seconds=0.1)
        body = {"messages": [{"role": "user", "content": "expire"}]}

        h, _ = cache.check(body)
        cache.complete(h, {"result": "old"})

        time.sleep(0.15)

        _, entry = cache.check(body)
        assert entry is None  # Expired

    def test_max_entries_eviction(self):
        cache = DedupCache(max_entries=3)

        for i in range(5):
            body = {"messages": [{"content": f"msg-{i}"}]}
            cache.check(body)

        assert cache.size <= 3

    def test_remove_on_error(self):
        cache = DedupCache()
        body = {"messages": [{"content": "error"}]}

        h, _ = cache.check(body)
        cache.remove(h)

        _, entry = cache.check(body)
        assert entry is None  # Removed, treated as new

    def test_stats_tracking(self):
        cache = DedupCache()
        body = {"messages": [{"content": "stats"}]}

        cache.check(body)  # miss
        assert cache.stats["misses"] == 1

        h, _ = cache.check(body)  # hit (pending)
        assert cache.stats["hits"] == 1

    def test_clear(self):
        cache = DedupCache()
        body = {"messages": [{"content": "clear"}]}
        cache.check(body)
        assert cache.size > 0
        cache.clear()
        assert cache.size == 0
