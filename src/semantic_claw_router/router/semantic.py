"""Semantic embedding classifier — fallback for ambiguous fast-path requests.

Uses sentence-transformers to embed requests and compare against pre-defined
anchor prompts for each complexity tier via cosine similarity.  Only loaded
when first needed (lazy initialization), keeping startup cost at zero.

Inspired by the signal-based classification architecture from
vLLM Semantic Router (https://github.com/vllm-project/semantic-router).

Model: all-MiniLM-L6-v2 (22M params, ~5-20 ms inference on CPU)
Requires: ``pip install semantic-claw-router[semantic]``
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from .types import ClassificationResult, ComplexityTier, ParsedRequest

if TYPE_CHECKING:
    import numpy as np

    from ..config import SemanticClassifierConfig

logger = logging.getLogger(__name__)


class _SemanticClassifierSingleton:
    """Thread-safe lazy singleton for the embedding model and anchor cache.

    The model is loaded exactly once on the first ambiguous request.
    Anchor prompts are embedded at that time and cached for the lifetime
    of the process.
    """

    _instance: _SemanticClassifierSingleton | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._model = None
        self._anchor_embeddings: dict[ComplexityTier, np.ndarray] | None = None
        self._config: SemanticClassifierConfig | None = None

    @classmethod
    def get_instance(cls, config: SemanticClassifierConfig) -> _SemanticClassifierSingleton:
        """Return the singleton, initializing on first call."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = cls()
                    inst._initialize(config)
                    cls._instance = inst
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing only)."""
        with cls._lock:
            cls._instance = None

    # ── Initialization ──────────────────────────────────────────────

    def _initialize(self, config: SemanticClassifierConfig) -> None:
        """Load the sentence-transformer model and pre-compute anchor embeddings."""
        from sentence_transformers import SentenceTransformer

        logger.info("Loading semantic classifier model: %s", config.model_name)
        self._model = SentenceTransformer(config.model_name)
        self._config = config

        # Pre-compute anchor embeddings for each tier
        self._anchor_embeddings = {}
        for tier in ComplexityTier:
            anchor_texts = config.anchors.get(tier.value, [])
            if anchor_texts:
                embeddings = self._model.encode(
                    anchor_texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                self._anchor_embeddings[tier] = embeddings

        total_anchors = sum(len(v) for v in self._anchor_embeddings.values())
        logger.info(
            "Semantic classifier ready: %d tiers, %d total anchors",
            len(self._anchor_embeddings),
            total_anchors,
        )

    # ── Classification ──────────────────────────────────────────────

    def classify(self, request: ParsedRequest) -> ClassificationResult:
        """Classify a request by embedding similarity to tier anchors.

        Algorithm:
        1. Embed the user's message text.
        2. Compute cosine similarity to every anchor embedding.
        3. For each tier, average the top-k similarities.
        4. Select the tier with the highest average score.
        5. Derive confidence from the margin between best and runner-up.
        """
        import numpy as np

        text = request.user_messages_text
        if not text.strip():
            return ClassificationResult(
                tier=ComplexityTier.SIMPLE,
                confidence=0.5,
                scores={},
                dominant_dimension="semantic_fallback",
            )

        # Embed the request
        query_embedding = self._model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]  # shape: (dim,)

        # Compute per-tier similarity scores
        tier_scores: dict[str, float] = {}
        top_k = self._config.top_k

        for tier, anchor_embs in self._anchor_embeddings.items():
            # Cosine similarity (embeddings are L2-normalized → dot product)
            similarities = anchor_embs @ query_embedding  # shape: (N,)
            k = min(top_k, len(similarities))
            top_k_sims = np.partition(similarities, -k)[-k:]
            tier_scores[tier.value] = float(np.mean(top_k_sims))

        # Select best tier
        sorted_tiers = sorted(tier_scores.items(), key=lambda x: x[1], reverse=True)
        best_tier_name, best_score = sorted_tiers[0]
        second_score = sorted_tiers[1][1] if len(sorted_tiers) > 1 else 0.0

        # Confidence from margin between best and runner-up
        # margin ~0.0 → conf 0.50, margin ~0.05 → 0.73, margin ~0.10 → 0.95
        margin = best_score - second_score
        confidence = min(0.5 + margin * 4.5, 0.99)
        confidence = max(confidence, 0.4)

        # Reuse fast-path agentic detection (embeddings aren't trained for this)
        from .fastpath import _score_agentic

        is_agentic = _score_agentic(request) > 0.5

        return ClassificationResult(
            tier=ComplexityTier(best_tier_name),
            confidence=confidence,
            scores=tier_scores,
            is_agentic=is_agentic,
            dominant_dimension="semantic_similarity",
        )


def classify_semantic(
    request: ParsedRequest,
    config: SemanticClassifierConfig,
) -> ClassificationResult | None:
    """Run the semantic embedding classifier.

    Returns a ``ClassificationResult``, or ``None`` if:
    - the semantic classifier is disabled in config,
    - ``sentence-transformers`` is not installed, or
    - an unexpected error occurs (logged as a warning).

    When ``None`` is returned the caller should fall back to the
    heuristic re-scoring approach.
    """
    if not config.enabled:
        return None

    try:
        instance = _SemanticClassifierSingleton.get_instance(config)
        return instance.classify(request)
    except ImportError:
        logger.warning(
            "sentence-transformers not installed; falling back to heuristic classifier. "
            "Install with: pip install semantic-claw-router[semantic]"
        )
        return None
    except Exception:
        logger.exception("Semantic classifier failed; falling back to heuristic")
        return None
