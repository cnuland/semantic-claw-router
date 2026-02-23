"""Tests for the decision engine."""

import pytest

from semantic_claw_router.config import RouterConfig
from semantic_claw_router.router.decision import DecisionEngine
from semantic_claw_router.router.types import (
    ClassificationResult,
    ComplexityTier,
    ModelBackend,
    RoutingDecisionSource,
)


@pytest.fixture
def config_with_models():
    config = RouterConfig()
    config.models = [
        ModelBackend(
            name="cheap-model",
            provider="vllm",
            endpoint="http://localhost:8000",
            context_window=32768,
            cost_per_million_input=0.0,
            cost_per_million_output=0.0,
        ),
        ModelBackend(
            name="expensive-model",
            provider="gemini",
            endpoint="https://api.example.com",
            api_key="test-key",
            context_window=1048576,
            cost_per_million_input=3.0,
            cost_per_million_output=15.0,
        ),
    ]
    config.default_tier_models = {
        "SIMPLE": "cheap-model",
        "MEDIUM": "cheap-model",
        "COMPLEX": "expensive-model",
        "REASONING": "expensive-model",
    }
    config.degradation.fallback_model = "cheap-model"
    return config


class TestDecisionEngine:
    def test_simple_routes_to_cheap(self, config_with_models):
        engine = DecisionEngine(config_with_models)
        classification = ClassificationResult(
            tier=ComplexityTier.SIMPLE, confidence=0.9
        )
        decision = engine.decide(classification)
        assert decision.target_model.name == "cheap-model"

    def test_complex_routes_to_expensive(self, config_with_models):
        engine = DecisionEngine(config_with_models)
        classification = ClassificationResult(
            tier=ComplexityTier.COMPLEX, confidence=0.9
        )
        decision = engine.decide(classification)
        assert decision.target_model.name == "expensive-model"

    def test_reasoning_routes_to_expensive(self, config_with_models):
        engine = DecisionEngine(config_with_models)
        classification = ClassificationResult(
            tier=ComplexityTier.REASONING, confidence=0.85
        )
        decision = engine.decide(classification)
        assert decision.target_model.name == "expensive-model"

    def test_decision_includes_metadata(self, config_with_models):
        engine = DecisionEngine(config_with_models)
        classification = ClassificationResult(
            tier=ComplexityTier.MEDIUM, confidence=0.8
        )
        decision = engine.decide(
            classification,
            source=RoutingDecisionSource.FAST_PATH,
            session_id="test-session",
        )
        assert decision.source == RoutingDecisionSource.FAST_PATH
        assert decision.session_id == "test-session"
        assert decision.confidence == 0.8

    def test_degradation_fallback(self, config_with_models):
        engine = DecisionEngine(config_with_models)
        decision = engine.decide_degraded(ComplexityTier.COMPLEX)
        assert decision is not None
        assert decision.target_model.name == "cheap-model"
        assert decision.source == RoutingDecisionSource.DEGRADATION
        assert decision.metadata.get("degraded") is True

    def test_no_degradation_when_unconfigured(self):
        config = RouterConfig()
        config.models = [
            ModelBackend(name="m1", provider="vllm", endpoint="http://localhost"),
        ]
        config.degradation.fallback_model = ""
        engine = DecisionEngine(config)
        assert engine.decide_degraded(ComplexityTier.COMPLEX) is None

    def test_fallback_chain_excludes_primary(self, config_with_models):
        engine = DecisionEngine(config_with_models)
        primary = config_with_models.models[1]  # expensive-model
        chain = engine.get_fallback_chain(primary)
        assert all(m.name != primary.name for m in chain)

    def test_fallback_chain_filters_by_context(self, config_with_models):
        engine = DecisionEngine(config_with_models)
        primary = config_with_models.models[1]
        # Request 50k tokens — cheap model only has 32k context
        chain = engine.get_fallback_chain(primary, context_tokens=50000)
        assert all(m.context_window >= 55000 for m in chain)  # 50k * 1.1

    def test_cost_estimation(self, config_with_models):
        engine = DecisionEngine(config_with_models)
        classification = ClassificationResult(
            tier=ComplexityTier.SIMPLE, confidence=0.9
        )
        decision = engine.decide(classification)
        # Cheap model has 0 cost
        assert decision.estimated_cost == 0.0

        classification_complex = ClassificationResult(
            tier=ComplexityTier.COMPLEX, confidence=0.9
        )
        decision_complex = engine.decide(classification_complex)
        # Expensive model should have non-zero cost
        assert decision_complex.estimated_cost > 0.0

    def test_no_models_raises(self):
        config = RouterConfig()
        engine = DecisionEngine(config)
        classification = ClassificationResult(
            tier=ComplexityTier.SIMPLE, confidence=0.9
        )
        with pytest.raises(ValueError, match="No models configured"):
            engine.decide(classification)
