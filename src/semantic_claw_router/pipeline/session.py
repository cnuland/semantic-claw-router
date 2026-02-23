"""Session pinning — keeps multi-turn conversations on the same model.

Inspired by ClawRouter (https://github.com/BlockRunAI/ClawRouter).
Original concept by BlockRun under MIT License.

Prevents jarring mid-conversation model switches by pinning
conversations to the initially selected model for a configurable TTL.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class SessionPin:
    """A pinned session entry."""

    model_name: str
    provider: str
    created_at: float
    last_used: float
    request_count: int = 1


class SessionTracker:
    """Thread-safe session-to-model pinning with TTL.

    Sessions are identified by a fingerprint derived from conversation
    content. The tracker maintains an LRU cache of session → model mappings.
    """

    def __init__(self, ttl_seconds: float = 3600.0, max_sessions: int = 10000):
        self._sessions: OrderedDict[str, SessionPin] = OrderedDict()
        self._lock = threading.Lock()
        self._ttl = ttl_seconds
        self._max_sessions = max_sessions
        self._stats = {"pins": 0, "hits": 0, "misses": 0, "expirations": 0}

    def fingerprint(
        self,
        messages: list[dict],
        api_key: str | None = None,
        custom_id: str | None = None,
    ) -> str:
        """Generate a conversation fingerprint for session tracking.

        Uses the first user message as the primary identity signal,
        combined with optional API key and custom session ID.
        """
        if custom_id:
            return custom_id

        parts = []
        # First user message content
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str) and content:
                    parts.append(content[:200])  # First 200 chars
                    break

        if api_key:
            parts.append(api_key[:16])  # Hash prefix only

        if not parts:
            return ""

        raw = "|".join(parts)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get_pin(self, session_id: str) -> SessionPin | None:
        """Look up an existing session pin.

        Returns the pin if found and not expired, None otherwise.
        """
        if not session_id:
            return None

        with self._lock:
            self._evict_expired()

            if session_id in self._sessions:
                pin = self._sessions[session_id]
                pin.last_used = time.time()
                pin.request_count += 1
                self._sessions.move_to_end(session_id)
                self._stats["hits"] += 1
                return pin

            self._stats["misses"] += 1
            return None

    def set_pin(self, session_id: str, model_name: str, provider: str) -> None:
        """Pin a session to a model."""
        if not session_id:
            return

        now = time.time()
        with self._lock:
            self._sessions[session_id] = SessionPin(
                model_name=model_name,
                provider=provider,
                created_at=now,
                last_used=now,
            )
            self._sessions.move_to_end(session_id)
            self._enforce_max_size()
            self._stats["pins"] += 1

    def remove_pin(self, session_id: str) -> None:
        """Remove a session pin (e.g., when model becomes unavailable)."""
        with self._lock:
            self._sessions.pop(session_id, None)

    def _evict_expired(self) -> None:
        now = time.time()
        expired = [
            k for k, v in self._sessions.items()
            if now - v.last_used > self._ttl
        ]
        for k in expired:
            del self._sessions[k]
            self._stats["expirations"] += 1

    def _enforce_max_size(self) -> None:
        while len(self._sessions) > self._max_sessions:
            self._sessions.popitem(last=False)

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._sessions)

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()
