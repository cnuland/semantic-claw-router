"""Tests for the semantic embedding fallback classifier.

Unit tests mock the embedding model to validate classification logic,
confidence calibration, and graceful degradation.  Integration tests
(marked @pytest.mark.semantic) use the real all-MiniLM-L6-v2 model
and are skipped when sentence-transformers is not installed.
"""

import pytest
from unittest.mock import patch, MagicMock

from semantic_claw_router.config import SemanticClassifierConfig
from semantic_claw_router.router.types import (
    ChatMessage,
    ClassificationResult,
    ComplexityTier,
    ParsedRequest,
)
from semantic_claw_router.router.semantic import (
    classify_semantic,
    _SemanticClassifierSingleton,
)


def _make_request(
    user_content: str,
    tools: list | None = None,
) -> ParsedRequest:
    """Helper to create a ParsedRequest for testing."""
    messages = [ChatMessage(role="user", content=user_content)]
    return ParsedRequest(
        model="test-model",
        messages=messages,
        tools=tools,
    )


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the classifier singleton before/after each test."""
    _SemanticClassifierSingleton.reset()
    yield
    _SemanticClassifierSingleton.reset()


# ── Unit Tests (always run, no sentence-transformers needed) ────────


class TestSemanticDisabled:
    """Tests for when the semantic classifier is disabled."""

    def test_returns_none_when_disabled(self):
        config = SemanticClassifierConfig(enabled=False)
        req = _make_request("What is Python?")
        result = classify_semantic(req, config)
        assert result is None


class TestGracefulDegradation:
    """Tests for graceful fallback when sentence-transformers is missing."""

    def test_returns_none_on_import_error(self):
        config = SemanticClassifierConfig(enabled=True)
        req = _make_request("What is Python?")

        with patch.object(
            _SemanticClassifierSingleton,
            "_initialize",
            side_effect=ImportError("No module named 'sentence_transformers'"),
        ):
            result = classify_semantic(req, config)
            assert result is None

    def test_returns_none_on_unexpected_error(self):
        config = SemanticClassifierConfig(enabled=True)
        req = _make_request("What is Python?")

        with patch.object(
            _SemanticClassifierSingleton,
            "get_instance",
            side_effect=RuntimeError("something broke"),
        ):
            result = classify_semantic(req, config)
            assert result is None


class TestClassificationLogic:
    """Tests for the core classification algorithm using mocked embeddings."""

    def _setup_mock_classifier(self, mock_encode_fn):
        """Create a classifier singleton with a mocked encode function."""
        import numpy as np

        instance = _SemanticClassifierSingleton()
        mock_model = MagicMock()
        mock_model.encode = mock_encode_fn
        instance._model = mock_model

        # Build anchor embeddings using the mock
        config = SemanticClassifierConfig()
        instance._config = config
        instance._anchor_embeddings = {}
        for tier in ComplexityTier:
            anchors = config.anchors.get(tier.value, [])
            if anchors:
                embeddings = mock_model.encode(
                    anchors,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                instance._anchor_embeddings[tier] = embeddings

        # Install as singleton
        _SemanticClassifierSingleton._instance = instance
        return instance

    def test_classifies_to_highest_similarity_tier(self):
        """Request should classify to the tier whose anchors are most similar."""
        import numpy as np

        # Create orthogonal unit vectors for each tier
        tier_vectors = {
            "SIMPLE": np.array([1.0, 0.0, 0.0, 0.0]),
            "MEDIUM": np.array([0.0, 1.0, 0.0, 0.0]),
            "COMPLEX": np.array([0.0, 0.0, 1.0, 0.0]),
            "REASONING": np.array([0.0, 0.0, 0.0, 1.0]),
        }

        call_count = [0]

        def mock_encode(texts, normalize_embeddings=True, show_progress_bar=False):
            """Return vectors that separate tiers clearly."""
            call_count[0] += 1
            config = SemanticClassifierConfig()
            # Determine which tier's anchors are being embedded
            for tier_name, anchors in config.anchors.items():
                if texts == anchors or (isinstance(texts, list) and set(texts) == set(anchors)):
                    return np.stack([tier_vectors[tier_name]] * len(texts))
            # For query text, return the SIMPLE vector
            return np.stack([tier_vectors["SIMPLE"]])

        self._setup_mock_classifier(mock_encode)

        req = _make_request("test query")
        config = SemanticClassifierConfig()
        result = classify_semantic(req, config)

        assert result is not None
        assert result.tier == ComplexityTier.SIMPLE
        assert result.dominant_dimension == "semantic_similarity"

    def test_empty_request_defaults_to_simple(self):
        """Empty request text should default to SIMPLE with low confidence."""
        import numpy as np

        def mock_encode(texts, **kwargs):
            return np.random.randn(len(texts), 4).astype(np.float32)

        self._setup_mock_classifier(mock_encode)

        req = _make_request("")
        config = SemanticClassifierConfig()
        result = classify_semantic(req, config)

        assert result is not None
        assert result.tier == ComplexityTier.SIMPLE
        assert result.confidence == 0.5

    def test_confidence_scales_with_margin(self):
        """Higher margin between best and runner-up → higher confidence."""
        import numpy as np

        # Directly set up the singleton with explicit tier embeddings
        instance = _SemanticClassifierSingleton()
        mock_model = MagicMock()
        # Query will return a vector aligned with REASONING
        mock_model.encode = lambda texts, **kw: np.stack(
            [np.array([0.0, 0.0, 0.0, 1.0])]
        )
        instance._model = mock_model
        instance._config = SemanticClassifierConfig()

        # Set anchor embeddings directly — REASONING aligned, others not
        instance._anchor_embeddings = {
            ComplexityTier.SIMPLE: np.stack(
                [np.array([0.3, 0.3, 0.3, 0.0])] * 6
            ),
            ComplexityTier.MEDIUM: np.stack(
                [np.array([0.3, 0.3, 0.3, 0.0])] * 7
            ),
            ComplexityTier.COMPLEX: np.stack(
                [np.array([0.3, 0.3, 0.3, 0.0])] * 7
            ),
            ComplexityTier.REASONING: np.stack(
                [np.array([0.0, 0.0, 0.0, 1.0])] * 7
            ),
        }
        _SemanticClassifierSingleton._instance = instance

        req = _make_request("prove by induction")
        config = SemanticClassifierConfig()
        result = classify_semantic(req, config)

        assert result is not None
        assert result.tier == ComplexityTier.REASONING
        assert result.confidence > 0.7  # Clear margin → high confidence

    def test_scores_contain_all_tiers(self):
        """Result scores should contain entries for all configured tiers."""
        import numpy as np

        def mock_encode(texts, **kwargs):
            return np.random.randn(len(texts) if isinstance(texts, list) else 1, 4).astype(
                np.float32
            )

        self._setup_mock_classifier(mock_encode)

        req = _make_request("some request")
        config = SemanticClassifierConfig()
        result = classify_semantic(req, config)

        assert result is not None
        for tier in ComplexityTier:
            assert tier.value in result.scores

    def test_confidence_clamped_between_bounds(self):
        """Confidence should always be in [0.4, 0.99]."""
        import numpy as np

        def mock_encode(texts, **kwargs):
            n = len(texts) if isinstance(texts, list) else 1
            return np.ones((n, 4), dtype=np.float32) * 0.25

        self._setup_mock_classifier(mock_encode)

        req = _make_request("test query")
        config = SemanticClassifierConfig()
        result = classify_semantic(req, config)

        assert result is not None
        assert 0.4 <= result.confidence <= 0.99


class TestSingletonBehavior:
    """Tests for the lazy singleton lifecycle."""

    def test_reset_clears_instance(self):
        """reset() should clear the singleton so next call re-initializes."""
        # Set a dummy instance
        _SemanticClassifierSingleton._instance = "dummy"
        _SemanticClassifierSingleton.reset()
        assert _SemanticClassifierSingleton._instance is None


class TestConfigDefaults:
    """Tests for default anchor configuration."""

    def test_default_anchors_cover_all_tiers(self):
        config = SemanticClassifierConfig()
        for tier in ComplexityTier:
            assert tier.value in config.anchors
            assert len(config.anchors[tier.value]) >= 5

    def test_default_model_name(self):
        config = SemanticClassifierConfig()
        assert config.model_name == "all-MiniLM-L6-v2"

    def test_default_top_k(self):
        config = SemanticClassifierConfig()
        assert config.top_k == 3


# ── Integration Tests (require sentence-transformers) ───────────────

try:
    import sentence_transformers

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


@pytest.mark.skipif(
    not HAS_SENTENCE_TRANSFORMERS,
    reason="sentence-transformers not installed (pip install semantic-claw-router[semantic])",
)
class TestSemanticClassifierIntegration:
    """Integration tests using the real all-MiniLM-L6-v2 model."""

    def test_simple_question_classifies_simple(self):
        config = SemanticClassifierConfig()
        req = _make_request("What is the capital of France?")
        result = classify_semantic(req, config)
        assert result is not None
        assert result.tier == ComplexityTier.SIMPLE
        assert result.confidence > 0.5

    def test_reasoning_proof_classifies_reasoning(self):
        config = SemanticClassifierConfig()
        req = _make_request(
            "Prove by contradiction that there are infinitely many prime numbers. "
            "Show each logical step of the derivation."
        )
        result = classify_semantic(req, config)
        assert result is not None
        assert result.tier == ComplexityTier.REASONING

    def test_code_task_classifies_medium(self):
        config = SemanticClassifierConfig()
        req = _make_request(
            "Write a Python function that takes a list of dictionaries and "
            "returns them sorted by a specified key, handling missing keys gracefully."
        )
        result = classify_semantic(req, config)
        assert result is not None
        assert result.tier in (ComplexityTier.SIMPLE, ComplexityTier.MEDIUM)

    def test_system_design_classifies_complex(self):
        config = SemanticClassifierConfig()
        req = _make_request(
            "Design a distributed event-driven architecture for processing "
            "financial transactions with exactly-once delivery guarantees, "
            "horizontal scaling, and multi-region failover."
        )
        result = classify_semantic(req, config)
        assert result is not None
        assert result.tier in (ComplexityTier.COMPLEX, ComplexityTier.REASONING)

    def test_result_has_all_tier_scores(self):
        config = SemanticClassifierConfig()
        req = _make_request("Explain how binary search works")
        result = classify_semantic(req, config)
        assert result is not None
        for tier in ComplexityTier:
            assert tier.value in result.scores

    def test_classification_latency(self):
        """Semantic classification should complete in <50ms after model load."""
        import time

        config = SemanticClassifierConfig()
        # Warm up (loads the model)
        classify_semantic(_make_request("warmup"), config)

        req = _make_request("Write a distributed database query optimizer")
        iterations = 20
        start = time.perf_counter()
        for _ in range(iterations):
            classify_semantic(req, config)
        elapsed_ms = (time.perf_counter() - start) * 1000
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 50, f"Semantic classification took {avg_ms:.1f}ms avg (target: <50ms)"

    def test_dominant_dimension_is_semantic(self):
        config = SemanticClassifierConfig()
        req = _make_request("What is a list comprehension?")
        result = classify_semantic(req, config)
        assert result is not None
        assert result.dominant_dimension == "semantic_similarity"
