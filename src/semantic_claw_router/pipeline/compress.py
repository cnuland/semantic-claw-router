"""Context auto-compression for large requests.

Inspired by ClawRouter (https://github.com/BlockRunAI/ClawRouter).
Original concept by BlockRun under MIT License.

Automatically compresses large contexts through whitespace normalization,
content deduplication, and JSON compaction before routing.
"""

from __future__ import annotations

import json
import re
from typing import Any


def compress_context(
    messages: list[dict[str, Any]],
    threshold_bytes: int = 184320,
    strategies: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compress message context if it exceeds the threshold.

    Returns:
        (compressed_messages, stats) where stats contains compression metrics.
    """
    if strategies is None:
        strategies = ["whitespace", "dedup", "json_compact"]

    # Measure original size
    original = json.dumps(messages)
    original_bytes = len(original.encode("utf-8"))

    if original_bytes < threshold_bytes:
        return messages, {
            "compressed": False,
            "original_bytes": original_bytes,
            "final_bytes": original_bytes,
            "savings_pct": 0.0,
        }

    compressed = [dict(m) for m in messages]  # Shallow copy

    for strategy in strategies:
        if strategy == "whitespace":
            compressed = _compress_whitespace(compressed)
        elif strategy == "dedup":
            compressed = _compress_dedup(compressed)
        elif strategy == "json_compact":
            compressed = _compress_json(compressed)

    final = json.dumps(compressed)
    final_bytes = len(final.encode("utf-8"))
    savings = (1 - final_bytes / original_bytes) * 100 if original_bytes > 0 else 0

    return compressed, {
        "compressed": True,
        "original_bytes": original_bytes,
        "final_bytes": final_bytes,
        "savings_pct": round(savings, 1),
        "strategies_applied": strategies,
    }


def _compress_whitespace(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize whitespace: collapse runs, normalize line endings."""
    result = []
    for msg in messages:
        new_msg = dict(msg)
        if isinstance(new_msg.get("content"), str):
            text = new_msg["content"]
            # Normalize line endings
            text = text.replace("\r\n", "\n")
            # Collapse runs of blank lines to single blank line
            text = re.sub(r"\n{3,}", "\n\n", text)
            # Collapse runs of spaces/tabs (not newlines)
            text = re.sub(r"[ \t]{2,}", " ", text)
            # Strip trailing whitespace per line
            text = re.sub(r"[ \t]+\n", "\n", text)
            new_msg["content"] = text.strip()
        result.append(new_msg)
    return result


def _compress_dedup(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate text blocks across messages.

    Especially useful after RAG injection where retrieved chunks
    may contain overlapping content.
    """
    seen_blocks: set[str] = set()
    result = []

    for msg in messages:
        new_msg = dict(msg)
        if isinstance(new_msg.get("content"), str):
            text = new_msg["content"]
            # Split into paragraphs and deduplicate
            paragraphs = text.split("\n\n")
            unique_paragraphs = []
            for para in paragraphs:
                normalized = para.strip().lower()
                # Only dedup substantial blocks (>100 chars)
                if len(normalized) > 100:
                    if normalized in seen_blocks:
                        continue
                    seen_blocks.add(normalized)
                unique_paragraphs.append(para)
            new_msg["content"] = "\n\n".join(unique_paragraphs)
        result.append(new_msg)
    return result


def _compress_json(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compact JSON/YAML blocks within message content."""
    result = []
    for msg in messages:
        new_msg = dict(msg)
        if isinstance(new_msg.get("content"), str):
            text = new_msg["content"]
            # Find JSON blocks in code fences and compact them
            text = re.sub(
                r"```(?:json)?\s*\n([\s\S]*?)\n```",
                _compact_json_block,
                text,
            )
            new_msg["content"] = text
        result.append(new_msg)
    return result


def _compact_json_block(match: re.Match) -> str:
    """Try to compact a JSON code block."""
    content = match.group(1)
    try:
        parsed = json.loads(content)
        compacted = json.dumps(parsed, separators=(",", ":"))
        return f"```json\n{compacted}\n```"
    except (json.JSONDecodeError, ValueError):
        return match.group(0)  # Return original if not valid JSON
