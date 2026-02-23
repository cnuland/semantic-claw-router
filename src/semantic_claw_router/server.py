"""Main HTTP proxy server for the semantic-claw-router.

This is the core entry point — an OpenAI-compatible HTTP server that
intercepts chat completion requests, runs them through the routing
pipeline, and forwards to the selected backend.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

import httpx

from .config import RouterConfig
from .observability.logging import setup_logging
from .observability.metrics import MetricsCollector, RequestMetrics
from .pipeline.compress import compress_context
from .pipeline.dedup import DedupCache
from .pipeline.session import SessionTracker
from .providers.gemini import GeminiProvider
from .providers.vllm import VLLMProvider
from .router.decision import DecisionEngine
from .router.fastpath import classify_fast_path
from .router.types import (
    ChatMessage,
    ClassificationResult,
    ComplexityTier,
    ModelBackend,
    ParsedRequest,
    RoutingDecisionSource,
    RoutingResponse,
)

logger = logging.getLogger(__name__)


class SemanticRouter:
    """Core routing engine that orchestrates the full request pipeline.

    Pipeline stages:
    1. Request parsing
    2. Deduplication check
    3. Fast-path classification
    4. Full classification (if fast-path is ambiguous)
    5. Decision engine (tier → model mapping)
    6. Session pinning check
    7. Context compression
    8. Provider routing
    9. Response processing (metrics, session update, dedup cache)
    """

    def __init__(self, config: RouterConfig):
        self.config = config
        self.decision_engine = DecisionEngine(config)
        self.dedup = DedupCache(
            max_entries=config.dedup.max_entries,
            window_seconds=config.dedup.window_seconds,
        ) if config.dedup.enabled else None
        self.sessions = SessionTracker(
            ttl_seconds=config.session.ttl_seconds,
            max_sessions=config.session.max_sessions,
        ) if config.session.enabled else None
        self.metrics = MetricsCollector()

        # Initialize providers
        self._providers: dict[str, Any] = {
            "vllm": VLLMProvider(timeout=config.request_timeout),
            "gemini": GeminiProvider(timeout=config.request_timeout),
        }

    def _parse_request(self, body: dict[str, Any]) -> ParsedRequest:
        """Parse an OpenAI-format chat completion request."""
        messages = []
        for m in body.get("messages", []):
            messages.append(ChatMessage(
                role=m.get("role", "user"),
                content=m.get("content"),
                tool_calls=m.get("tool_calls"),
                tool_call_id=m.get("tool_call_id"),
                name=m.get("name"),
            ))

        return ParsedRequest(
            model=body.get("model", ""),
            messages=messages,
            tools=body.get("tools"),
            stream=body.get("stream", False),
            temperature=body.get("temperature"),
            max_tokens=body.get("max_tokens"),
            top_p=body.get("top_p"),
            raw_body=body,
        )

    async def route_request(
        self,
        body: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> RoutingResponse:
        """Run the full routing pipeline on a request."""
        request_id = str(uuid.uuid4())[:8]
        start = time.monotonic()
        headers = headers or {}

        # ── Stage 1: Parse request ──
        parsed = self._parse_request(body)

        # ── Stage 2: Dedup check ──
        dedup_hash = None
        if self.dedup:
            dedup_hash, existing = self.dedup.check(body)
            if existing and existing.status.value == "completed" and existing.response:
                logger.info("Dedup hit for request %s", request_id)
                self.metrics.record_request(RequestMetrics(
                    timestamp=time.time(),
                    tier="cached",
                    model="dedup",
                    source="dedup_cache",
                    latency_ms=(time.monotonic() - start) * 1000,
                    status_code=200,
                    was_deduped=True,
                ))
                return RoutingResponse(
                    status_code=200,
                    headers={"content-type": "application/json",
                             "x-scr-dedup": "true"},
                    body=existing.response,
                )

        # ── Stage 3: Session pinning check ──
        session_id = None
        pinned_model = None
        if self.sessions:
            raw_messages = body.get("messages", [])
            session_id = self.sessions.fingerprint(
                raw_messages,
                api_key=headers.get("authorization"),
                custom_id=headers.get("x-session-id"),
            )
            pin = self.sessions.get_pin(session_id)
            if pin:
                pinned_model = self.config.get_model(pin.model_name)
                if pinned_model:
                    logger.info(
                        "Session pin hit: %s → %s", session_id, pin.model_name
                    )

        # ── Stage 4: Classification ──
        if pinned_model:
            # Skip classification — use pinned model
            classification = ClassificationResult(
                tier=ComplexityTier.MEDIUM,
                confidence=1.0,
                is_agentic=False,
                dominant_dimension="session_pin",
            )
            source = RoutingDecisionSource.SESSION_PIN
            target_model = pinned_model
        else:
            # Run fast-path classifier
            classification = classify_fast_path(parsed, self.config.fast_path)

            if classification is None:
                # Fast-path was ambiguous — run "full" classification
                # For POC, we use a simplified full classifier that defaults to MEDIUM
                classification = self._full_classify(parsed)
                source = RoutingDecisionSource.FULL_CLASSIFICATION
            else:
                source = RoutingDecisionSource.FAST_PATH

            # ── Stage 5: Decision engine ──
            decision = self.decision_engine.decide(classification, source, session_id)
            target_model = decision.target_model

        # ── Stage 6: Context compression ──
        compression_stats = {"compressed": False}
        if self.config.compression.enabled:
            raw_messages = body.get("messages", [])
            compressed_messages, compression_stats = compress_context(
                raw_messages,
                threshold_bytes=self.config.compression.threshold_bytes,
                strategies=self.config.compression.strategies,
            )
            if compression_stats["compressed"]:
                body = dict(body)
                body["messages"] = compressed_messages
                logger.info(
                    "Compressed context: %d → %d bytes (%.1f%% savings)",
                    compression_stats["original_bytes"],
                    compression_stats["final_bytes"],
                    compression_stats["savings_pct"],
                )

        # ── Stage 7: Route to provider ──
        provider = self._get_provider(target_model)
        response = await provider.chat_completion(target_model, body)

        # ── Stage 8: Fallback on error ──
        if response.status_code >= 500 or response.status_code == 429:
            logger.warning(
                "Primary model %s returned %d, trying fallback",
                target_model.name, response.status_code,
            )
            fallback_chain = self.decision_engine.get_fallback_chain(
                target_model, parsed.estimated_tokens
            )
            for fallback in fallback_chain:
                fb_provider = self._get_provider(fallback)
                response = await fb_provider.chat_completion(fallback, body)
                if response.status_code < 400:
                    target_model = fallback
                    source = RoutingDecisionSource.DEGRADATION
                    logger.info("Fallback succeeded with %s", fallback.name)
                    break

            # Ultimate fallback: degradation model
            if response.status_code >= 400 and self.config.degradation.enabled:
                deg_decision = self.decision_engine.decide_degraded(
                    classification.tier
                )
                if deg_decision:
                    deg_provider = self._get_provider(deg_decision.target_model)
                    response = await deg_provider.chat_completion(
                        deg_decision.target_model, body
                    )
                    if response.status_code < 400:
                        target_model = deg_decision.target_model
                        source = RoutingDecisionSource.DEGRADATION

        # ── Stage 9: Post-processing ──
        latency = (time.monotonic() - start) * 1000

        # Update session pin
        if self.sessions and session_id and response.status_code == 200:
            self.sessions.set_pin(session_id, target_model.name, target_model.provider)

        # Update dedup cache
        if self.dedup and dedup_hash and response.status_code == 200 and response.body:
            self.dedup.complete(dedup_hash, response.body)
        elif self.dedup and dedup_hash and response.status_code >= 400:
            self.dedup.remove(dedup_hash)

        # Record metrics
        tokens_in = response.tokens_used.get("prompt_tokens", 0)
        tokens_out = response.tokens_used.get("completion_tokens", 0)
        self.metrics.record_request(RequestMetrics(
            timestamp=time.time(),
            tier=classification.tier.value,
            model=target_model.name,
            source=source.value,
            latency_ms=latency,
            status_code=response.status_code,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            estimated_cost=self.decision_engine._estimate_cost(
                target_model, classification
            ),
            was_deduped=False,
            was_compressed=compression_stats.get("compressed", False),
            compression_savings_pct=compression_stats.get("savings_pct", 0),
            was_degraded=(source == RoutingDecisionSource.DEGRADATION),
        ))

        # Add routing metadata headers
        response.headers["x-scr-model"] = target_model.name
        response.headers["x-scr-tier"] = classification.tier.value
        response.headers["x-scr-source"] = source.value
        response.headers["x-scr-latency-ms"] = str(round(latency, 1))
        response.headers["x-scr-request-id"] = request_id
        if classification.dominant_dimension:
            response.headers["x-scr-dominant-signal"] = classification.dominant_dimension
        if source == RoutingDecisionSource.DEGRADATION:
            response.headers["x-scr-degraded"] = "true"

        response.routing_decision = self.decision_engine.decide(
            classification, source, session_id
        )
        response.latency_ms = latency

        logger.info(
            "Routed request %s: tier=%s model=%s source=%s latency=%.0fms",
            request_id, classification.tier.value, target_model.name,
            source.value, latency,
        )

        return response

    def _full_classify(self, parsed: ParsedRequest) -> ClassificationResult:
        """Fallback classification when fast-path is ambiguous.

        For the POC, this uses a simplified heuristic. In production,
        this would invoke neural BERT classifiers via Candle.
        """
        # Use a simpler version of the fast-path with lower threshold
        from .router.fastpath import DIMENSION_SCORERS

        scores = {}
        weighted_sum = 0.0
        weights = self.config.fast_path.weights

        for dim_name, scorer in DIMENSION_SCORERS.items():
            weight = weights.get(dim_name, 0.0)
            raw_score = scorer(parsed)
            scores[dim_name] = raw_score
            weighted_sum += weight * raw_score

        boundaries = self.config.fast_path.tier_boundaries
        if weighted_sum < boundaries.get("simple", 0.0):
            tier = ComplexityTier.SIMPLE
        elif weighted_sum < boundaries.get("medium", 0.3):
            tier = ComplexityTier.MEDIUM
        elif weighted_sum < boundaries.get("complex", 0.5):
            tier = ComplexityTier.COMPLEX
        else:
            tier = ComplexityTier.REASONING

        dominant = max(scores, key=lambda k: abs(scores[k] * weights.get(k, 0)))

        return ClassificationResult(
            tier=tier,
            confidence=0.6,  # Lower confidence since fast-path was ambiguous
            scores=scores,
            is_agentic=scores.get("agentic_task", 0) > 0.5,
            dominant_dimension=dominant,
        )

    def _get_provider(self, model: ModelBackend):
        """Get the provider instance for a model backend."""
        provider_type = model.provider
        if provider_type in self._providers:
            return self._providers[provider_type]
        # Default to vllm (OpenAI-compatible)
        return self._providers["vllm"]

    async def close(self) -> None:
        """Clean up provider connections."""
        for provider in self._providers.values():
            await provider.close()


# ── HTTP Server ──────────────────────────────────────────────────────

class RouterHTTPHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the semantic-claw-router.

    Provides an OpenAI-compatible API:
    - POST /v1/chat/completions — Route a chat completion request
    - GET  /v1/models — List configured models
    - GET  /health — Health check
    - GET  /metrics — Router metrics
    """

    router: SemanticRouter  # Set by the server factory
    loop: asyncio.AbstractEventLoop

    def log_message(self, format, *args):
        """Suppress default request logging (we use structured logging)."""
        pass

    def do_GET(self):
        if self.path == "/v1/models":
            self._handle_models()
        elif self.path == "/health":
            self._handle_health()
        elif self.path == "/metrics":
            self._handle_metrics()
        else:
            self._send_json(404, {"error": {"message": "Not found"}})

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            self._handle_chat_completion()
        else:
            self._send_json(404, {"error": {"message": "Not found"}})

    def _handle_chat_completion(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(content_length)

        try:
            body = json.loads(body_bytes)
        except json.JSONDecodeError:
            self._send_json(400, {"error": {"message": "Invalid JSON"}})
            return

        headers = {k.lower(): v for k, v in self.headers.items()}

        # Run the async routing pipeline
        future = asyncio.run_coroutine_threadsafe(
            self.router.route_request(body, headers),
            self.loop,
        )
        try:
            response = future.result(timeout=self.router.config.request_timeout + 5)
        except TimeoutError:
            self._send_json(504, {"error": {"message": "Request timeout"}})
            return
        except Exception as e:
            logger.exception("Error routing request")
            self._send_json(500, {"error": {"message": str(e)}})
            return

        self._send_json(
            response.status_code,
            response.body or {},
            extra_headers=response.headers,
        )

    def _handle_models(self):
        models = [
            {
                "id": m.name,
                "object": "model",
                "owned_by": m.provider,
                "context_window": m.context_window,
            }
            for m in self.router.config.models
        ]
        self._send_json(200, {"object": "list", "data": models})

    def _handle_health(self):
        self._send_json(200, {"status": "healthy", "version": "0.1.0"})

    def _handle_metrics(self):
        summary = self.router.metrics.get_summary()
        if self.router.dedup:
            summary["dedup_cache_size"] = self.router.dedup.size
            summary["dedup_stats"] = self.router.dedup.stats
        if self.router.sessions:
            summary["session_cache_size"] = self.router.sessions.size
            summary["session_stats"] = self.router.sessions.stats
        self._send_json(200, summary)

    def _send_json(
        self,
        status: int,
        body: dict,
        extra_headers: dict[str, str] | None = None,
    ):
        response_bytes = json.dumps(body, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        if extra_headers:
            for k, v in extra_headers.items():
                if k.lower().startswith("x-scr-"):
                    self.send_header(k, v)
        self.end_headers()
        self.wfile.write(response_bytes)


def create_server(config: RouterConfig) -> tuple[HTTPServer, SemanticRouter]:
    """Create an HTTP server with the routing engine.

    Returns (server, router) tuple.
    """
    setup_logging(
        level=config.observability.log_level,
        fmt=config.observability.log_format,
    )

    router = SemanticRouter(config)

    # Create event loop for async operations
    loop = asyncio.new_event_loop()

    import threading
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()

    # Configure handler class
    handler = type("Handler", (RouterHTTPHandler,), {
        "router": router,
        "loop": loop,
    })

    server = HTTPServer((config.host, config.port), handler)
    logger.info(
        "Semantic Claw Router starting on %s:%d with %d models",
        config.host, config.port, len(config.models),
    )
    for m in config.models:
        logger.info("  Model: %s (%s) → %s", m.name, m.provider, m.endpoint)

    return server, router
