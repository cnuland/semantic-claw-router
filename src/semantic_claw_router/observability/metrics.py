"""Metrics tracking for router observability."""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class RequestMetrics:
    """Metrics for a single request."""

    timestamp: float
    tier: str
    model: str
    source: str  # fast_path, full_classification, session_pin, etc.
    latency_ms: float
    status_code: int
    tokens_in: int = 0
    tokens_out: int = 0
    estimated_cost: float = 0.0
    was_deduped: bool = False
    was_compressed: bool = False
    compression_savings_pct: float = 0.0
    was_degraded: bool = False


class MetricsCollector:
    """Thread-safe metrics collection and aggregation.

    Collects per-request metrics and provides aggregate statistics.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._requests: list[RequestMetrics] = []
        self._counters: dict[str, int] = defaultdict(int)
        self._start_time = time.time()

    def record_request(self, metrics: RequestMetrics) -> None:
        with self._lock:
            self._requests.append(metrics)
            self._counters["total_requests"] += 1
            self._counters[f"tier_{metrics.tier}"] += 1
            self._counters[f"model_{metrics.model}"] += 1
            self._counters[f"source_{metrics.source}"] += 1
            self._counters[f"status_{metrics.status_code}"] += 1
            if metrics.was_deduped:
                self._counters["dedup_hits"] += 1
            if metrics.was_compressed:
                self._counters["compressions"] += 1
            if metrics.was_degraded:
                self._counters["degradations"] += 1

    def get_summary(self) -> dict:
        """Get aggregate metrics summary."""
        with self._lock:
            if not self._requests:
                return {"total_requests": 0, "uptime_seconds": time.time() - self._start_time}

            latencies = [r.latency_ms for r in self._requests]
            costs = [r.estimated_cost for r in self._requests]
            total_tokens_in = sum(r.tokens_in for r in self._requests)
            total_tokens_out = sum(r.tokens_out for r in self._requests)

            return {
                "total_requests": len(self._requests),
                "uptime_seconds": round(time.time() - self._start_time, 1),
                "counters": dict(self._counters),
                "latency": {
                    "mean_ms": round(sum(latencies) / len(latencies), 1),
                    "min_ms": round(min(latencies), 1),
                    "max_ms": round(max(latencies), 1),
                    "p50_ms": round(sorted(latencies)[len(latencies) // 2], 1),
                    "p99_ms": round(
                        sorted(latencies)[int(len(latencies) * 0.99)], 1
                    ),
                },
                "tokens": {
                    "total_input": total_tokens_in,
                    "total_output": total_tokens_out,
                },
                "cost": {
                    "total_estimated": round(sum(costs), 6),
                },
                "tier_distribution": {
                    k.replace("tier_", ""): v
                    for k, v in self._counters.items()
                    if k.startswith("tier_")
                },
                "model_distribution": {
                    k.replace("model_", ""): v
                    for k, v in self._counters.items()
                    if k.startswith("model_")
                },
                "routing_source_distribution": {
                    k.replace("source_", ""): v
                    for k, v in self._counters.items()
                    if k.startswith("source_")
                },
            }

    def clear(self) -> None:
        with self._lock:
            self._requests.clear()
            self._counters.clear()
            self._start_time = time.time()
