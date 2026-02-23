"""Tests for configuration loading."""

import tempfile
from pathlib import Path

import pytest
import yaml

from semantic_claw_router.config import RouterConfig
from semantic_claw_router.router.types import ComplexityTier


class TestConfigLoading:
    def test_load_from_yaml(self, tmp_path):
        config_data = {
            "host": "127.0.0.1",
            "port": 9090,
            "models": [
                {
                    "name": "test-model",
                    "provider": "vllm",
                    "endpoint": "http://localhost:8000",
                    "context_window": 32768,
                }
            ],
            "default_tier_models": {
                "SIMPLE": "test-model",
                "MEDIUM": "test-model",
            },
            "fast_path": {
                "enabled": True,
                "confidence_threshold": 0.8,
            },
            "dedup": {
                "enabled": False,
            },
        }
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config_data))

        config = RouterConfig.from_yaml(str(config_path))
        assert config.host == "127.0.0.1"
        assert config.port == 9090
        assert len(config.models) == 1
        assert config.models[0].name == "test-model"
        assert config.fast_path.confidence_threshold == 0.8
        assert config.dedup.enabled is False

    def test_default_values(self):
        config = RouterConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8080
        assert config.fast_path.enabled is True
        assert config.dedup.enabled is True
        assert config.session.enabled is True
        assert config.compression.enabled is True

    def test_get_model(self):
        from semantic_claw_router.router.types import ModelBackend
        config = RouterConfig()
        config.models = [
            ModelBackend(name="m1", provider="vllm", endpoint="http://a"),
            ModelBackend(name="m2", provider="gemini", endpoint="http://b"),
        ]
        assert config.get_model("m1").name == "m1"
        assert config.get_model("m2").provider == "gemini"
        assert config.get_model("m3") is None

    def test_get_model_for_tier(self):
        from semantic_claw_router.router.types import ModelBackend
        config = RouterConfig()
        config.models = [
            ModelBackend(name="cheap", provider="vllm", endpoint="http://a"),
            ModelBackend(name="expensive", provider="gemini", endpoint="http://b"),
        ]
        config.default_tier_models = {
            "SIMPLE": "cheap",
            "REASONING": "expensive",
        }
        assert config.get_model_for_tier(ComplexityTier.SIMPLE).name == "cheap"
        assert config.get_model_for_tier(ComplexityTier.REASONING).name == "expensive"
        # Unmapped tier falls back to first model
        assert config.get_model_for_tier(ComplexityTier.MEDIUM).name == "cheap"

    def test_missing_config_file(self):
        with pytest.raises(FileNotFoundError):
            RouterConfig.from_yaml("/nonexistent/path.yaml")

    def test_custom_weights(self, tmp_path):
        config_data = {
            "models": [],
            "fast_path": {
                "weights": {
                    "code_presence": 0.50,
                    "reasoning_markers": 0.01,
                },
            },
        }
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config_data))

        config = RouterConfig.from_yaml(str(config_path))
        assert config.fast_path.weights["code_presence"] == 0.50
        assert config.fast_path.weights["reasoning_markers"] == 0.01
        # Other weights should keep defaults
        assert config.fast_path.weights["technical_terms"] == 0.10
