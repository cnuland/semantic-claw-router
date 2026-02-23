"""Tests for the fast-path pre-classifier.

Tests the 14-dimension weighted scoring system, tier mapping,
confidence calibration, and special override behaviors.
"""

import time

import pytest

from semantic_claw_router.config import FastPathConfig
from semantic_claw_router.router.fastpath import (
    DIMENSION_SCORERS,
    classify_fast_path,
    _sigmoid_confidence,
)
from semantic_claw_router.router.types import (
    ChatMessage,
    ClassificationResult,
    ComplexityTier,
    ParsedRequest,
)


def _make_request(
    user_content: str,
    system_content: str | None = None,
    tools: list | None = None,
) -> ParsedRequest:
    """Helper to create a ParsedRequest for testing."""
    messages = []
    if system_content:
        messages.append(ChatMessage(role="system", content=system_content))
    messages.append(ChatMessage(role="user", content=user_content))
    return ParsedRequest(
        model="test-model",
        messages=messages,
        tools=tools,
    )


@pytest.fixture
def default_config() -> FastPathConfig:
    return FastPathConfig()


# ── Tier Classification Tests ────────────────────────────────────────

class TestTierClassification:
    def test_simple_question(self, default_config):
        """Simple factual questions should classify as SIMPLE."""
        req = _make_request("What is Python?")
        result = classify_fast_path(req, default_config)
        assert result is not None
        assert result.tier == ComplexityTier.SIMPLE

    def test_translation_is_simple(self, default_config):
        """Translation requests should classify as SIMPLE."""
        req = _make_request("Translate 'hello' to French")
        result = classify_fast_path(req, default_config)
        assert result is not None
        assert result.tier == ComplexityTier.SIMPLE

    def test_code_generation_medium_to_complex(self, default_config):
        """Code generation with moderate complexity → MEDIUM or COMPLEX."""
        req = _make_request(
            "Write a Python function that reads a CSV file and returns "
            "the average of a column. Use the csv module."
        )
        result = classify_fast_path(req, default_config)
        assert result is not None
        assert result.tier in (ComplexityTier.MEDIUM, ComplexityTier.COMPLEX)

    def test_complex_system_design(self, default_config):
        """Multi-step system design → at least MEDIUM (likely COMPLEX+)."""
        req = _make_request(
            "Design a distributed system architecture for a real-time "
            "chat application. First, design the database schema. Then, "
            "implement the WebSocket server. Ensure horizontal scalability "
            "and implement message ordering guarantees. The system must "
            "handle at least 100k concurrent connections."
        )
        result = classify_fast_path(req, default_config)
        assert result is not None
        # Should be at least MEDIUM — may reach COMPLEX/REASONING with
        # enough signals. The exact tier depends on weight calibration.
        assert result.tier in (
            ComplexityTier.MEDIUM, ComplexityTier.COMPLEX, ComplexityTier.REASONING
        )
        # Should score high on technical_terms and imperative_verbs
        assert result.scores["technical_terms"] > 0.5
        assert result.scores["imperative_verbs"] > 0.5

    def test_reasoning_with_proof(self, default_config):
        """Mathematical proof requests should force REASONING tier."""
        req = _make_request(
            "Prove by induction that the sum of first n natural numbers "
            "equals n*(n+1)/2. Show each step of the derivation."
        )
        result = classify_fast_path(req, default_config)
        assert result is not None
        assert result.tier == ComplexityTier.REASONING

    def test_multiple_reasoning_keywords_force_reasoning(self, default_config):
        """2+ reasoning keywords in user prompt should force REASONING."""
        req = _make_request(
            "Prove this theorem using mathematical induction and derive "
            "the contradiction."
        )
        result = classify_fast_path(req, default_config)
        assert result is not None
        assert result.tier == ComplexityTier.REASONING
        assert result.confidence >= 0.85


# ── Dimension Scorer Tests ───────────────────────────────────────────

class TestDimensionScorers:
    def test_code_presence_detects_code(self):
        req = _make_request(
            "def function(x): return x + 1\nclass MyClass: pass\nimport os"
        )
        score = DIMENSION_SCORERS["code_presence"](req)
        assert score > 0.3

    def test_code_presence_no_code(self):
        req = _make_request("Tell me about the weather today")
        score = DIMENSION_SCORERS["code_presence"](req)
        assert score < 0.1

    def test_reasoning_markers_user_only(self):
        """Reasoning markers should only count in user messages, not system."""
        req_user = _make_request("Prove this theorem step by step")
        req_system = _make_request(
            "Hello",
            system_content="You must prove and derive theorems step by step"
        )
        score_user = DIMENSION_SCORERS["reasoning_markers"](req_user)
        score_system = DIMENSION_SCORERS["reasoning_markers"](req_system)
        assert score_user > score_system

    def test_simple_indicators_negative(self):
        """Simple indicators should produce negative scores."""
        req = _make_request("What is the capital of France?")
        score = DIMENSION_SCORERS["simple_indicators"](req)
        assert score <= 0.0

    def test_multi_step_detection(self):
        req = _make_request(
            "First, read the file. Then, parse the JSON. Next, filter "
            "the results. After that, write the output."
        )
        score = DIMENSION_SCORERS["multi_step_patterns"](req)
        assert score > 0.3

    def test_agentic_with_tools(self):
        """Requests with tools array should score high on agentic."""
        req = _make_request(
            "Read the file and fix the bug",
            tools=[{"type": "function", "function": {"name": "read_file"}}],
        )
        score = DIMENSION_SCORERS["agentic_task"](req)
        assert score >= 0.6

    def test_agentic_without_tools(self):
        req = _make_request("What is Python?")
        score = DIMENSION_SCORERS["agentic_task"](req)
        assert score < 0.3

    def test_domain_specificity(self):
        req = _make_request(
            "What is the eigenvalue decomposition of a Hilbert space operator?"
        )
        score = DIMENSION_SCORERS["domain_specificity"](req)
        assert score > 0.2

    def test_output_format_json(self):
        req = _make_request("Return the results as JSON with a schema")
        score = DIMENSION_SCORERS["output_format"](req)
        assert score > 0.2

    def test_token_count_short(self):
        req = _make_request("Hi")
        score = DIMENSION_SCORERS["token_count"](req)
        assert score < 0.0  # Short → negative

    def test_token_count_long(self):
        req = _make_request("word " * 1000)
        score = DIMENSION_SCORERS["token_count"](req)
        assert score > 0.5  # Long → positive


# ── Confidence Calibration Tests ─────────────────────────────────────

class TestConfidenceCalibration:
    def test_high_confidence_far_from_boundary(self):
        """Scores far from tier boundaries should have high confidence."""
        boundaries = {"simple": 0.0, "medium": 0.3, "complex": 0.5}
        conf = _sigmoid_confidence(-0.5, boundaries)  # Far from 0.0
        assert conf > 0.7

    def test_low_confidence_near_boundary(self):
        """Scores near tier boundaries should have low confidence."""
        boundaries = {"simple": 0.0, "medium": 0.3, "complex": 0.5}
        conf = _sigmoid_confidence(0.01, boundaries)  # Very close to 0.0
        assert conf < 0.6

    def test_confidence_between_0_and_1(self):
        """Confidence should always be in [0, 1]."""
        boundaries = {"simple": 0.0, "medium": 0.3, "complex": 0.5}
        for score in [-1.0, -0.5, 0.0, 0.15, 0.3, 0.5, 0.8, 1.0]:
            conf = _sigmoid_confidence(score, boundaries)
            assert 0.0 <= conf <= 1.0


# ── Config Tests ─────────────────────────────────────────────────────

class TestFastPathConfig:
    def test_disabled_returns_none(self):
        config = FastPathConfig(enabled=False)
        req = _make_request("What is Python?")
        assert classify_fast_path(req, config) is None

    def test_low_threshold_classifies_more(self):
        """Lower confidence threshold → more requests classified by fast-path."""
        config_low = FastPathConfig(confidence_threshold=0.3)
        config_high = FastPathConfig(confidence_threshold=0.9)

        # Ambiguous request
        req = _make_request("Help me with this code thing")
        result_low = classify_fast_path(req, config_low)
        result_high = classify_fast_path(req, config_high)

        # Low threshold should classify it, high might return None
        assert result_low is not None or result_high is None

    def test_all_dimensions_have_scorers(self):
        """Every configured weight dimension must have a scorer function."""
        config = FastPathConfig()
        for dim in config.weights:
            assert dim in DIMENSION_SCORERS, f"Missing scorer for {dim}"


# ── Performance Tests ────────────────────────────────────────────────

class TestFastPathPerformance:
    @pytest.mark.benchmark
    def test_classification_speed(self, default_config):
        """Fast-path classification should complete in <1ms."""
        req = _make_request(
            "Write a Python function to sort a list using quicksort algorithm"
        )

        # Warm up
        classify_fast_path(req, default_config)

        # Benchmark
        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            classify_fast_path(req, default_config)
        elapsed = (time.perf_counter() - start) * 1000  # ms

        avg_ms = elapsed / iterations
        assert avg_ms < 1.0, f"Classification took {avg_ms:.3f}ms (target: <1ms)"

    @pytest.mark.benchmark
    def test_empty_request_speed(self, default_config):
        """Even edge cases should be fast."""
        req = _make_request("")

        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            classify_fast_path(req, default_config)
        elapsed = (time.perf_counter() - start) * 1000

        avg_ms = elapsed / iterations
        assert avg_ms < 1.0
