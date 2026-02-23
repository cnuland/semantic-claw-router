"""Core type definitions for the routing system."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ComplexityTier(str, Enum):
    """Request complexity tiers — maps request difficulty to model capability needs."""

    SIMPLE = "SIMPLE"
    MEDIUM = "MEDIUM"
    COMPLEX = "COMPLEX"
    REASONING = "REASONING"


class RoutingDecisionSource(str, Enum):
    """How the routing decision was made."""

    FAST_PATH = "fast_path"
    FULL_CLASSIFICATION = "full_classification"
    SEMANTIC_CLASSIFICATION = "semantic_classification"
    SESSION_PIN = "session_pin"
    DEDUP_CACHE = "dedup_cache"
    OVERRIDE = "override"
    DEGRADATION = "degradation"


@dataclass
class ClassificationResult:
    """Output of the fast-path or full classifier."""

    tier: ComplexityTier
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)
    is_agentic: bool = False
    dominant_dimension: str = ""


@dataclass
class ChatMessage:
    """A single message in a chat completion request."""

    role: str
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None


@dataclass
class ParsedRequest:
    """Parsed OpenAI-compatible chat completion request."""

    model: str
    messages: list[ChatMessage]
    tools: list[dict[str, Any]] | None = None
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    raw_body: dict[str, Any] = field(default_factory=dict)

    @property
    def user_messages_text(self) -> str:
        """Concatenated text of all user messages."""
        return " ".join(
            m.content for m in self.messages if m.role == "user" and m.content
        )

    @property
    def system_messages_text(self) -> str:
        """Concatenated text of all system messages."""
        return " ".join(
            m.content for m in self.messages if m.role == "system" and m.content
        )

    @property
    def all_messages_text(self) -> str:
        """All message content concatenated."""
        return " ".join(m.content for m in self.messages if m.content)

    @property
    def has_tools(self) -> bool:
        return bool(self.tools)

    @property
    def estimated_tokens(self) -> int:
        """Rough token estimate (1 token ≈ 4 chars)."""
        return len(self.all_messages_text) // 4


@dataclass
class ModelBackend:
    """A model backend that can serve requests."""

    name: str
    provider: str  # "vllm", "gemini", "openai", etc.
    endpoint: str
    api_key: str | None = None
    context_window: int = 32768
    cost_per_million_input: float = 0.0
    cost_per_million_output: float = 0.0
    supports_tools: bool = True
    supports_streaming: bool = True


@dataclass
class RoutingDecision:
    """The final routing decision for a request."""

    target_model: ModelBackend
    tier: ComplexityTier
    source: RoutingDecisionSource
    confidence: float
    classification: ClassificationResult | None = None
    session_id: str | None = None
    estimated_cost: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingResponse:
    """The response from the backend, wrapped with routing metadata."""

    status_code: int
    headers: dict[str, str]
    body: dict[str, Any] | None = None
    raw_body: bytes = b""
    is_streaming: bool = False
    routing_decision: RoutingDecision | None = None
    latency_ms: float = 0.0
    tokens_used: dict[str, int] = field(default_factory=dict)
