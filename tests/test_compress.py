"""Tests for context auto-compression."""

import json

import pytest

from semantic_claw_router.pipeline.compress import compress_context


class TestCompressionThreshold:
    def test_below_threshold_no_compression(self):
        messages = [{"role": "user", "content": "Hello"}]
        result, stats = compress_context(messages, threshold_bytes=1000)
        assert not stats["compressed"]
        assert result == messages

    def test_above_threshold_triggers_compression(self):
        # Create a large message
        large_content = "word " * 50000  # ~250KB
        messages = [{"role": "user", "content": large_content}]
        result, stats = compress_context(messages, threshold_bytes=1000)
        assert stats["compressed"]
        assert stats["final_bytes"] <= stats["original_bytes"]


class TestWhitespaceCompression:
    def test_collapses_blank_lines(self):
        messages = [{"role": "user", "content": "line1\n\n\n\n\nline2"}]
        result, _ = compress_context(messages, threshold_bytes=0, strategies=["whitespace"])
        assert "\n\n\n" not in result[0]["content"]
        assert "line1" in result[0]["content"]
        assert "line2" in result[0]["content"]

    def test_collapses_spaces(self):
        messages = [{"role": "user", "content": "word    word     word"}]
        result, _ = compress_context(messages, threshold_bytes=0, strategies=["whitespace"])
        assert "    " not in result[0]["content"]

    def test_strips_trailing_whitespace(self):
        messages = [{"role": "user", "content": "line1   \nline2  \n"}]
        result, _ = compress_context(messages, threshold_bytes=0, strategies=["whitespace"])
        content = result[0]["content"]
        for line in content.split("\n"):
            assert line == line.rstrip()


class TestDedupCompression:
    def test_removes_duplicate_paragraphs(self):
        # Simulate RAG injection with duplicate chunks
        paragraph = "This is a long paragraph that contains important information. " * 10
        messages = [
            {"role": "system", "content": f"Context:\n\n{paragraph}\n\n{paragraph}\n\n{paragraph}"},
            {"role": "user", "content": "What does the context say?"},
        ]
        result, _ = compress_context(messages, threshold_bytes=0, strategies=["dedup"])
        # Should have fewer occurrences of the paragraph
        original_count = messages[0]["content"].count(paragraph.strip().lower()[:50])
        result_text = result[0]["content"].lower()
        # The duplicate should be removed
        assert len(result_text) < len(messages[0]["content"])

    def test_keeps_short_paragraphs(self):
        """Short paragraphs (< 100 chars) should not be deduped."""
        messages = [{"role": "user", "content": "Short.\n\nShort.\n\nShort."}]
        result, _ = compress_context(messages, threshold_bytes=0, strategies=["dedup"])
        assert result[0]["content"].count("Short.") == 3


class TestJsonCompaction:
    def test_compacts_json_blocks(self):
        json_content = json.dumps({"key": "value", "nested": {"a": 1}}, indent=4)
        messages = [{"role": "user", "content": f"Here:\n```json\n{json_content}\n```"}]
        result, _ = compress_context(messages, threshold_bytes=0, strategies=["json_compact"])
        # Should be compacted (no indentation)
        assert "    " not in result[0]["content"]
        # But still valid JSON
        content = result[0]["content"]
        json_start = content.index("```json\n") + len("```json\n")
        json_end = content.index("\n```", json_start)
        parsed = json.loads(content[json_start:json_end])
        assert parsed == {"key": "value", "nested": {"a": 1}}

    def test_invalid_json_left_unchanged(self):
        messages = [{"role": "user", "content": "```json\nnot valid json\n```"}]
        result, _ = compress_context(messages, threshold_bytes=0, strategies=["json_compact"])
        assert "not valid json" in result[0]["content"]


class TestCompressionStats:
    def test_stats_include_savings(self):
        large_content = "word   " * 50000
        messages = [{"role": "user", "content": large_content}]
        _, stats = compress_context(messages, threshold_bytes=0)
        assert stats["compressed"]
        assert stats["savings_pct"] > 0
        assert stats["original_bytes"] > stats["final_bytes"]
        assert "strategies_applied" in stats

    def test_no_compression_stats(self):
        messages = [{"role": "user", "content": "small"}]
        _, stats = compress_context(messages, threshold_bytes=999999)
        assert not stats["compressed"]
        assert stats["savings_pct"] == 0.0
