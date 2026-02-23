"""Integration tests against live endpoints.

These tests require:
- Qwen3 model running on OpenShift
- Gemini API key configured

Run with: pytest tests/test_integration.py -m integration
"""

import asyncio
import json

import pytest

from semantic_claw_router.config import RouterConfig
from semantic_claw_router.router.types import ModelBackend
from semantic_claw_router.providers.vllm import VLLMProvider
from semantic_claw_router.providers.gemini import GeminiProvider
from semantic_claw_router.server import SemanticRouter


QWEN_ENDPOINT = (
    "https://qwen3-coder-next-llm-serving"
    ".instructlab-ai-7407a08c5ee477a3d3ffccce3a9f8665-0000"
    ".eu-gb.containers.appdomain.cloud"
)
GEMINI_API_KEY = "***REMOVED***"


def _make_config() -> RouterConfig:
    config = RouterConfig()
    config.models = [
        ModelBackend(
            name="qwen3-coder-next",
            provider="vllm",
            endpoint=QWEN_ENDPOINT,
            context_window=32768,
        ),
        ModelBackend(
            name="gemini-2.5-flash",
            provider="gemini",
            endpoint="https://generativelanguage.googleapis.com/v1beta",
            api_key=GEMINI_API_KEY,
            context_window=1048576,
            cost_per_million_input=0.15,
            cost_per_million_output=0.60,
        ),
    ]
    config.default_tier_models = {
        "SIMPLE": "qwen3-coder-next",
        "MEDIUM": "qwen3-coder-next",
        "COMPLEX": "gemini-2.5-flash",
        "REASONING": "gemini-2.5-flash",
    }
    config.degradation.fallback_model = "qwen3-coder-next"
    return config


@pytest.fixture
def config():
    return _make_config()


@pytest.fixture
def router(config):
    return SemanticRouter(config)


# ── Provider Tests ───────────────────────────────────────────────────

@pytest.mark.integration
class TestVLLMProvider:
    @pytest.mark.asyncio
    async def test_chat_completion(self):
        provider = VLLMProvider(timeout=30.0)
        model = ModelBackend(
            name="qwen3-coder-next",
            provider="vllm",
            endpoint=QWEN_ENDPOINT,
        )
        body = {
            "messages": [{"role": "user", "content": "Say 'hello' and nothing else."}],
            "max_tokens": 100,
        }
        response = await provider.chat_completion(model, body)
        await provider.close()

        assert response.status_code == 200
        assert response.body is not None
        assert "choices" in response.body
        assert len(response.body["choices"]) > 0
        assert response.latency_ms > 0

    @pytest.mark.asyncio
    async def test_health_check(self):
        provider = VLLMProvider(timeout=10.0)
        model = ModelBackend(
            name="qwen3-coder-next",
            provider="vllm",
            endpoint=QWEN_ENDPOINT,
        )
        healthy = await provider.health_check(model)
        await provider.close()
        assert healthy is True


@pytest.mark.integration
class TestGeminiProvider:
    @pytest.mark.asyncio
    async def test_chat_completion(self):
        provider = GeminiProvider(timeout=30.0)
        model = ModelBackend(
            name="gemini-2.5-flash",
            provider="gemini",
            endpoint="https://generativelanguage.googleapis.com/v1beta",
            api_key=GEMINI_API_KEY,
        )
        body = {
            "messages": [{"role": "user", "content": "Say 'hello' and nothing else."}],
            "max_tokens": 100,
        }
        response = await provider.chat_completion(model, body)
        await provider.close()

        assert response.status_code == 200
        assert response.body is not None
        assert "choices" in response.body
        content = response.body["choices"][0]["message"]["content"]
        assert len(content) > 0

    @pytest.mark.asyncio
    async def test_system_message_translation(self):
        provider = GeminiProvider(timeout=30.0)
        model = ModelBackend(
            name="gemini-2.5-flash",
            provider="gemini",
            endpoint="https://generativelanguage.googleapis.com/v1beta",
            api_key=GEMINI_API_KEY,
        )
        body = {
            "messages": [
                {"role": "system", "content": "You only respond with the word 'pong'."},
                {"role": "user", "content": "ping"},
            ],
            "max_tokens": 100,
        }
        response = await provider.chat_completion(model, body)
        await provider.close()

        assert response.status_code == 200
        content = response.body["choices"][0]["message"]["content"].lower()
        assert "pong" in content


# ── Full Router Pipeline Tests ───────────────────────────────────────

@pytest.mark.integration
class TestRouterPipeline:
    @pytest.mark.asyncio
    async def test_simple_request_routes_to_qwen(self, router):
        """Simple requests should route to the local Qwen3 model."""
        body = {
            "model": "auto",
            "messages": [{"role": "user", "content": "What is Python?"}],
            "max_tokens": 50,
        }
        response = await router.route_request(body)
        await router.close()

        assert response.status_code == 200
        assert response.headers.get("x-scr-tier") == "SIMPLE"
        assert response.headers.get("x-scr-model") == "qwen3-coder-next"

    @pytest.mark.asyncio
    async def test_complex_request_routes_to_gemini(self, router):
        """Complex reasoning requests should route to Gemini."""
        body = {
            "model": "auto",
            "messages": [{
                "role": "user",
                "content": (
                    "Prove by mathematical induction that for all n >= 1, "
                    "the sum 1 + 2 + ... + n = n(n+1)/2. Show the base case, "
                    "inductive hypothesis, and derive the inductive step."
                ),
            }],
            "max_tokens": 500,
        }
        response = await router.route_request(body)
        await router.close()

        assert response.status_code == 200
        assert response.headers.get("x-scr-tier") == "REASONING"
        assert response.headers.get("x-scr-model") == "gemini-2.5-flash"

    @pytest.mark.asyncio
    async def test_coding_request_classification(self, router):
        """Code generation with moderate complexity should route appropriately."""
        body = {
            "model": "auto",
            "messages": [{
                "role": "user",
                "content": "Write a Python function to reverse a string.",
            }],
            "max_tokens": 200,
        }
        response = await router.route_request(body)
        await router.close()

        assert response.status_code == 200
        assert "choices" in response.body

    @pytest.mark.asyncio
    async def test_dedup_returns_cached(self, router):
        """Duplicate requests should return cached responses."""
        body = {
            "model": "auto",
            "messages": [{"role": "user", "content": "Hello dedup test"}],
            "max_tokens": 20,
        }

        # First request
        resp1 = await router.route_request(body)
        assert resp1.status_code == 200

        # Duplicate request
        resp2 = await router.route_request(body)
        assert resp2.status_code == 200
        assert resp2.headers.get("x-scr-dedup") == "true"

        await router.close()

    @pytest.mark.asyncio
    async def test_session_pinning(self, router):
        """Multi-turn conversation should pin to the same model."""
        body1 = {
            "model": "auto",
            "messages": [
                {"role": "user", "content": "My name is Alice. Remember that."}
            ],
            "max_tokens": 50,
        }
        resp1 = await router.route_request(body1, {"x-session-id": "pin-test"})

        body2 = {
            "model": "auto",
            "messages": [
                {"role": "user", "content": "My name is Alice. Remember that."},
                {"role": "assistant", "content": "I'll remember that!"},
                {"role": "user", "content": "What is my name?"},
            ],
            "max_tokens": 50,
        }
        resp2 = await router.route_request(body2, {"x-session-id": "pin-test"})

        await router.close()

        # Both should use the same model due to session pinning
        assert resp1.headers.get("x-scr-model") == resp2.headers.get("x-scr-model")
        assert resp2.headers.get("x-scr-source") == "session_pin"

    @pytest.mark.asyncio
    async def test_metrics_populated(self, router):
        """Metrics should be populated after requests."""
        body = {
            "model": "auto",
            "messages": [{"role": "user", "content": "Test metrics"}],
            "max_tokens": 100,
        }
        await router.route_request(body)
        await router.close()

        summary = router.metrics.get_summary()
        assert summary["total_requests"] == 1
        assert summary["latency"]["mean_ms"] > 0
