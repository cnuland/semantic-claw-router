"""Decision engine — selects the target model based on classification results.

This module maps classification results (tier + signals) to concrete
model backends. It supports tier-based routing, fallback chains, and
degradation policies.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .types import (
    ClassificationResult,
    ComplexityTier,
    ModelBackend,
    RoutingDecision,
    RoutingDecisionSource,
)

if TYPE_CHECKING:
    from ..config import RouterConfig

logger = logging.getLogger(__name__)


class DecisionEngine:
    """Evaluates classification results and selects target models."""

    def __init__(self, config: RouterConfig):
        self.config = config
        self._model_index: dict[str, ModelBackend] = {
            m.name: m for m in config.models
        }

    def decide(
        self,
        classification: ClassificationResult,
        source: RoutingDecisionSource = RoutingDecisionSource.FAST_PATH,
        session_id: str | None = None,
    ) -> RoutingDecision:
        """Make a routing decision based on classification results.

        Looks up the configured model for the classified tier. Falls back
        to the first available model if no tier mapping exists.
        """
        target = self._select_model_for_tier(classification.tier)

        if target is None:
            logger.warning(
                "No model configured for tier %s, using first available",
                classification.tier,
            )
            target = self.config.models[0] if self.config.models else None

        if target is None:
            raise ValueError("No models configured in router")

        # Estimate cost
        estimated_cost = self._estimate_cost(target, classification)

        return RoutingDecision(
            target_model=target,
            tier=classification.tier,
            source=source,
            confidence=classification.confidence,
            classification=classification,
            session_id=session_id,
            estimated_cost=estimated_cost,
        )

    def decide_degraded(self, original_tier: ComplexityTier) -> RoutingDecision | None:
        """Select the degradation fallback model."""
        fallback = self.config.get_fallback_model()
        if fallback is None:
            return None

        return RoutingDecision(
            target_model=fallback,
            tier=original_tier,
            source=RoutingDecisionSource.DEGRADATION,
            confidence=1.0,
            metadata={"degraded": True, "original_tier": original_tier.value},
        )

    def _select_model_for_tier(self, tier: ComplexityTier) -> ModelBackend | None:
        """Look up the model configured for a complexity tier."""
        model_name = self.config.default_tier_models.get(tier.value)
        if model_name and model_name in self._model_index:
            return self._model_index[model_name]
        return self.config.get_model_for_tier(tier)

    def _estimate_cost(
        self, model: ModelBackend, classification: ClassificationResult
    ) -> float:
        """Rough cost estimate based on model pricing and expected tokens."""
        # Assume ~500 input tokens and ~500 output tokens for a typical request
        input_cost = (500 / 1_000_000) * model.cost_per_million_input
        output_cost = (500 / 1_000_000) * model.cost_per_million_output
        return input_cost + output_cost

    def get_fallback_chain(
        self, primary: ModelBackend, context_tokens: int = 0
    ) -> list[ModelBackend]:
        """Get ordered fallback models filtered by context window capacity."""
        fallbacks = []
        for m in self.config.models:
            if m.name == primary.name:
                continue
            # 10% buffer on context window
            if context_tokens > 0 and m.context_window < context_tokens * 1.1:
                continue
            fallbacks.append(m)
        # Sort by cost (cheapest first for fallback)
        fallbacks.sort(key=lambda m: m.cost_per_million_input)
        return fallbacks
