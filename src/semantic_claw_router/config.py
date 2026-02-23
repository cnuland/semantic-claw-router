"""Configuration model for the semantic-claw-router."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .router.types import ComplexityTier, ModelBackend


@dataclass
class FastPathConfig:
    """Configuration for the fast-path pre-classifier."""

    enabled: bool = True
    confidence_threshold: float = 0.7
    weights: dict[str, float] = field(default_factory=lambda: {
        "token_count": 0.08,
        "code_presence": 0.15,
        "reasoning_markers": 0.18,
        "technical_terms": 0.10,
        "creative_markers": 0.05,
        "simple_indicators": 0.08,
        "multi_step_patterns": 0.12,
        "question_complexity": 0.05,
        "imperative_verbs": 0.03,
        "constraint_indicators": 0.04,
        "output_format": 0.03,
        "reference_complexity": 0.02,
        "negation_complexity": 0.01,
        "domain_specificity": 0.02,
        "agentic_task": 0.04,
    })
    tier_boundaries: dict[str, float] = field(default_factory=lambda: {
        "simple": 0.0,
        "medium": 0.3,
        "complex": 0.5,
    })
    tier_to_model: dict[str, str] = field(default_factory=dict)


@dataclass
class DedupConfig:
    """Configuration for request deduplication."""

    enabled: bool = True
    window_seconds: float = 30.0
    max_entries: int = 10000


@dataclass
class SessionConfig:
    """Configuration for session pinning."""

    enabled: bool = True
    ttl_seconds: float = 3600.0
    max_sessions: int = 10000


@dataclass
class CompressionConfig:
    """Configuration for context auto-compression."""

    enabled: bool = True
    threshold_bytes: int = 184320  # 180KB
    strategies: list[str] = field(default_factory=lambda: [
        "whitespace", "dedup", "json_compact"
    ])


@dataclass
class DegradationConfig:
    """Configuration for graceful degradation."""

    enabled: bool = True
    fallback_model: str = ""
    triggers: list[str] = field(default_factory=lambda: [
        "provider_error", "rate_limit", "timeout"
    ])


@dataclass
class ObservabilityConfig:
    """Configuration for observability."""

    log_level: str = "INFO"
    log_format: str = "json"
    metrics_enabled: bool = True


@dataclass
class RouterConfig:
    """Top-level router configuration."""

    host: str = "0.0.0.0"
    port: int = 8080
    models: list[ModelBackend] = field(default_factory=list)
    fast_path: FastPathConfig = field(default_factory=FastPathConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    degradation: DegradationConfig = field(default_factory=DegradationConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    default_tier_models: dict[str, str] = field(default_factory=dict)
    request_timeout: float = 120.0

    @classmethod
    def from_yaml(cls, path: str | Path) -> RouterConfig:
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> RouterConfig:
        config = cls()
        config.host = data.get("host", config.host)
        config.port = data.get("port", config.port)
        config.request_timeout = data.get("request_timeout", config.request_timeout)

        for m in data.get("models", []):
            config.models.append(ModelBackend(
                name=m["name"],
                provider=m["provider"],
                endpoint=m["endpoint"],
                api_key=m.get("api_key"),
                context_window=m.get("context_window", 32768),
                cost_per_million_input=m.get("cost_per_million_input", 0.0),
                cost_per_million_output=m.get("cost_per_million_output", 0.0),
                supports_tools=m.get("supports_tools", True),
                supports_streaming=m.get("supports_streaming", True),
            ))

        fp = data.get("fast_path", {})
        if fp:
            config.fast_path.enabled = fp.get("enabled", True)
            config.fast_path.confidence_threshold = fp.get(
                "confidence_threshold", 0.7
            )
            if "weights" in fp:
                config.fast_path.weights.update(fp["weights"])
            if "tier_boundaries" in fp:
                config.fast_path.tier_boundaries.update(fp["tier_boundaries"])
            if "tier_to_model" in fp:
                config.fast_path.tier_to_model.update(fp["tier_to_model"])

        dd = data.get("dedup", {})
        if dd:
            config.dedup.enabled = dd.get("enabled", True)
            config.dedup.window_seconds = dd.get("window_seconds", 30.0)
            config.dedup.max_entries = dd.get("max_entries", 10000)

        sess = data.get("session", {})
        if sess:
            config.session.enabled = sess.get("enabled", True)
            config.session.ttl_seconds = sess.get("ttl_seconds", 3600.0)

        comp = data.get("compression", {})
        if comp:
            config.compression.enabled = comp.get("enabled", True)
            config.compression.threshold_bytes = comp.get(
                "threshold_bytes", 184320
            )

        deg = data.get("degradation", {})
        if deg:
            config.degradation.enabled = deg.get("enabled", True)
            config.degradation.fallback_model = deg.get("fallback_model", "")

        config.default_tier_models = data.get("default_tier_models", {})

        return config

    def get_model(self, name: str) -> ModelBackend | None:
        """Look up a model backend by name."""
        for m in self.models:
            if m.name == name:
                return m
        return None

    def get_model_for_tier(self, tier: ComplexityTier) -> ModelBackend | None:
        """Get the configured model for a complexity tier."""
        model_name = self.default_tier_models.get(tier.value)
        if model_name:
            return self.get_model(model_name)
        # Fallback: return first model
        return self.models[0] if self.models else None

    def get_fallback_model(self) -> ModelBackend | None:
        """Get the degradation fallback model."""
        if self.degradation.fallback_model:
            return self.get_model(self.degradation.fallback_model)
        return None
