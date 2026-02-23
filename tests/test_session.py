"""Tests for session pinning."""

import time

import pytest

from semantic_claw_router.pipeline.session import SessionTracker


class TestFingerprinting:
    def test_same_first_message_same_fingerprint(self):
        tracker = SessionTracker()
        msgs1 = [{"role": "user", "content": "Hello, help me with code"}]
        msgs2 = [{"role": "user", "content": "Hello, help me with code"}]
        fp1 = tracker.fingerprint(msgs1)
        fp2 = tracker.fingerprint(msgs2)
        assert fp1 == fp2

    def test_different_messages_different_fingerprint(self):
        tracker = SessionTracker()
        msgs1 = [{"role": "user", "content": "Hello"}]
        msgs2 = [{"role": "user", "content": "Goodbye"}]
        fp1 = tracker.fingerprint(msgs1)
        fp2 = tracker.fingerprint(msgs2)
        assert fp1 != fp2

    def test_custom_id_overrides(self):
        tracker = SessionTracker()
        msgs = [{"role": "user", "content": "Hello"}]
        fp = tracker.fingerprint(msgs, custom_id="my-session-123")
        assert fp == "my-session-123"

    def test_api_key_differentiates(self):
        tracker = SessionTracker()
        msgs = [{"role": "user", "content": "Hello"}]
        fp1 = tracker.fingerprint(msgs, api_key="key-1")
        fp2 = tracker.fingerprint(msgs, api_key="key-2")
        assert fp1 != fp2

    def test_empty_messages_empty_fingerprint(self):
        tracker = SessionTracker()
        fp = tracker.fingerprint([])
        assert fp == ""

    def test_system_only_messages_empty_fingerprint(self):
        tracker = SessionTracker()
        msgs = [{"role": "system", "content": "You are helpful"}]
        fp = tracker.fingerprint(msgs)
        assert fp == ""


class TestSessionPinning:
    def test_pin_and_retrieve(self):
        tracker = SessionTracker()
        tracker.set_pin("session-1", "gpt-4o", "openai")

        pin = tracker.get_pin("session-1")
        assert pin is not None
        assert pin.model_name == "gpt-4o"
        assert pin.provider == "openai"

    def test_miss_returns_none(self):
        tracker = SessionTracker()
        pin = tracker.get_pin("nonexistent")
        assert pin is None

    def test_ttl_expiration(self):
        tracker = SessionTracker(ttl_seconds=0.1)
        tracker.set_pin("session-1", "model-a", "vllm")

        pin = tracker.get_pin("session-1")
        assert pin is not None

        time.sleep(0.15)

        pin = tracker.get_pin("session-1")
        assert pin is None  # Expired

    def test_request_count_increments(self):
        tracker = SessionTracker()
        tracker.set_pin("session-1", "model-a", "vllm")

        tracker.get_pin("session-1")  # count=2
        tracker.get_pin("session-1")  # count=3
        pin = tracker.get_pin("session-1")  # count=4
        assert pin is not None
        assert pin.request_count == 4

    def test_last_used_updates(self):
        tracker = SessionTracker()
        tracker.set_pin("session-1", "model-a", "vllm")

        pin1 = tracker.get_pin("session-1")
        first_used = pin1.last_used
        time.sleep(0.02)
        pin2 = tracker.get_pin("session-1")
        assert pin2.last_used > first_used

    def test_max_sessions_eviction(self):
        tracker = SessionTracker(max_sessions=3)

        for i in range(5):
            tracker.set_pin(f"session-{i}", f"model-{i}", "vllm")

        assert tracker.size <= 3

    def test_remove_pin(self):
        tracker = SessionTracker()
        tracker.set_pin("session-1", "model-a", "vllm")
        tracker.remove_pin("session-1")

        pin = tracker.get_pin("session-1")
        assert pin is None

    def test_empty_session_id_ignored(self):
        tracker = SessionTracker()
        tracker.set_pin("", "model-a", "vllm")
        pin = tracker.get_pin("")
        assert pin is None

    def test_stats_tracking(self):
        tracker = SessionTracker()
        tracker.set_pin("s1", "m1", "p1")
        assert tracker.stats["pins"] == 1

        tracker.get_pin("s1")
        assert tracker.stats["hits"] == 1

        tracker.get_pin("nonexistent")
        assert tracker.stats["misses"] == 1

    def test_clear(self):
        tracker = SessionTracker()
        tracker.set_pin("s1", "m1", "p1")
        assert tracker.size == 1
        tracker.clear()
        assert tracker.size == 0
