"""Request deduplication filter.

Inspired by ClawRouter (https://github.com/BlockRunAI/ClawRouter).
Original concept by BlockRun under MIT License.

Prevents duplicate inference when clients retry after timeouts by
hashing canonicalized request content and caching results.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DedupStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"


@dataclass
class DedupEntry:
    """A cached dedup entry."""

    status: DedupStatus
    timestamp: float
    response: dict[str, Any] | None = None


# Fields to strip before hashing (these change between retries)
_STRIP_FIELDS = frozenset({
    "timestamp", "request_id", "x-request-id", "trace_id",
    "idempotency_key", "client_request_id",
})


def canonicalize_request(body: dict[str, Any]) -> str:
    """Canonicalize a request body for consistent hashing.

    1. Remove volatile fields (timestamps, request IDs)
    2. Sort keys recursively
    3. Serialize deterministically
    """
    cleaned = _strip_volatile(body)
    return json.dumps(cleaned, sort_keys=True, separators=(",", ":"))


def _strip_volatile(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            k: _strip_volatile(v)
            for k, v in sorted(obj.items())
            if k.lower() not in _STRIP_FIELDS
        }
    if isinstance(obj, list):
        return [_strip_volatile(item) for item in obj]
    return obj


def hash_request(canonical: str) -> str:
    """Compute SHA-256 hash of canonicalized request."""
    return hashlib.sha256(canonical.encode()).hexdigest()


class DedupCache:
    """Thread-safe LRU deduplication cache with TTL eviction.

    Stores request hashes → (status, response) pairs.
    - PENDING: request is being processed
    - COMPLETED: response is cached and can be returned immediately
    """

    def __init__(self, max_entries: int = 10000, window_seconds: float = 30.0):
        self._cache: OrderedDict[str, DedupEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._max_entries = max_entries
        self._window_seconds = window_seconds
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}

    def check(self, request_body: dict[str, Any]) -> tuple[str, DedupEntry | None]:
        """Check if a request is a duplicate.

        Returns:
            (hash, entry) — entry is None if this is a new request.
            If entry.status == COMPLETED, the caller should return the cached response.
            If entry.status == PENDING, another identical request is in-flight.
        """
        canonical = canonicalize_request(request_body)
        h = hash_request(canonical)

        with self._lock:
            self._evict_expired()

            if h in self._cache:
                entry = self._cache[h]
                self._cache.move_to_end(h)
                self._stats["hits"] += 1
                return h, entry

            # Mark as pending
            self._cache[h] = DedupEntry(
                status=DedupStatus.PENDING,
                timestamp=time.time(),
            )
            self._enforce_max_size()
            self._stats["misses"] += 1
            return h, None

    def complete(self, h: str, response: dict[str, Any]) -> None:
        """Mark a request as completed and cache the response."""
        with self._lock:
            if h in self._cache:
                self._cache[h] = DedupEntry(
                    status=DedupStatus.COMPLETED,
                    timestamp=time.time(),
                    response=response,
                )
                self._cache.move_to_end(h)

    def remove(self, h: str) -> None:
        """Remove an entry (e.g., on error — don't cache failures)."""
        with self._lock:
            self._cache.pop(h, None)

    def _evict_expired(self) -> None:
        now = time.time()
        expired = [
            k for k, v in self._cache.items()
            if now - v.timestamp > self._window_seconds
        ]
        for k in expired:
            del self._cache[k]
            self._stats["evictions"] += 1

    def _enforce_max_size(self) -> None:
        while len(self._cache) > self._max_entries:
            self._cache.popitem(last=False)
            self._stats["evictions"] += 1

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._cache)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
