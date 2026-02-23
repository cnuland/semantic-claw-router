# Benchmarks

Detailed performance benchmarks for the Semantic Claw Router. All benchmarks run on Apple Silicon, Python 3.11, against live model backends (vLLM on OpenShift + Google Gemini).

---

## Routing Overhead

| Metric | Target | Measured |
|--------|--------|----------|
| Fast-path classifier (per request) | < 1 ms | **27–46 μs** (0.03–0.05 ms) |
| Full pipeline overhead (no network) | < 5 ms | **34 μs** (0.034 ms) |
| End-to-end overhead (with live models) | < 5 ms | **0.1 ms** |
| Dedup cache lookup | < 0.1 ms | **< 0.01 ms** |
| Session pin lookup | < 0.1 ms | **< 0.01 ms** |

**The router adds 0.1 ms of overhead to each request.** Against a typical 200–800 ms inference call, that's **0.05% overhead**.

## Classifier Performance (per prompt, 1000 iterations)

| Prompt | Tier | mean | p50 | p99 |
|--------|------|------|-----|-----|
| What is a Python decorator? | SIMPLE | 27 μs | 27 μs | 34 μs |
| Convert hex #FF5733 to RGB | SIMPLE | 28 μs | 28 μs | 34 μs |
| Hello! | SIMPLE | 12 μs | 12 μs | 15 μs |
| Write a merge sort function | MEDIUM | 35 μs | 35 μs | 41 μs |
| Review this class for best practices | MEDIUM | 32 μs | 32 μs | 41 μs |
| Prove Dijkstra by induction | REASONING | 39 μs | 38 μs | 48 μs |
| Prove comparison sort lower bound | REASONING | 35 μs | 35 μs | 40 μs |

## Full Pipeline Overhead (5000 iterations)

Parse + Dedup + Session + Classify + Decide (no network I/O):

```
  mean:  34.2 μs  (0.034 ms)
  p50:   34.1 μs  (0.034 ms)
  p99:   43.8 μs  (0.044 ms)
```

## End-to-End with Live Models

| Prompt | Tier | Model | Total | Inference | Overhead |
|--------|------|-------|-------|-----------|----------|
| What is Python? | SIMPLE | qwen3-coder-next | 217 ms | 217 ms | 0.1 ms |
| Hello! | SIMPLE | qwen3-coder-next | 128 ms | 128 ms | 0.1 ms |
| Convert #FF5733 to RGB | SIMPLE | qwen3-coder-next | 220 ms | 220 ms | 0.1 ms |
| Write a binary search function | MEDIUM | qwen3-coder-next | 222 ms | 221 ms | 0.1 ms |
| Fix: result=0; ... result*=i | MEDIUM | qwen3-coder-next | 221 ms | 221 ms | 0.1 ms |
| Write pytest tests | MEDIUM | qwen3-coder-next | 219 ms | 218 ms | 0.1 ms |
| Design REST API with JWT | MEDIUM | qwen3-coder-next | 217 ms | 217 ms | 0.1 ms |
| Prove Dijkstra by induction | REASONING | gemini-2.5-flash | 846 ms | 846 ms | 0.2 ms |
| Prove sort lower bound | MEDIUM | qwen3-coder-next | 222 ms | 222 ms | 0.1 ms |
| What is a list comprehension? | SIMPLE | qwen3-coder-next | 218 ms | 218 ms | 0.1 ms |

## Model Usage Breakdown

```
  qwen3-coder-next (local)     9/10 ( 90%)  ██████████████████  FREE (self-hosted)
  gemini-2.5-flash (cloud)     1/10 ( 10%)  ██                  PAID (cloud API)
```

**90% of requests routed to the free local model.** Only requests requiring genuine mathematical reasoning hit the paid cloud API.

## Classification Accuracy

Against 21 diverse prompts (definition lookups, code generation, system design, formal proofs, agentic tool-use):

- **86% exact tier match** with the heuristic fast-path classifier alone
- Borderline cases (score near 0.3 MEDIUM/COMPLEX boundary) account for most misses
- Production deployments would add a neural BERT classifier as a second stage for ambiguous cases

## Semantic Classifier Latency

When the fast-path is ambiguous, the semantic embedding classifier fires:

| Metric | Measured |
|--------|----------|
| Model load (first request only) | ~1-3 s (downloads + loads `all-MiniLM-L6-v2`) |
| Per-request embedding + similarity | ~5-20 ms on CPU |
| Frequency | ~14% of requests (fast-path handles 86%) |

The semantic classifier only fires for ambiguous requests. For the 86% handled by the fast-path, there is zero additional cost.

## Test Coverage

| Module | Tests | What's Covered |
|--------|-------|----------------|
| Fast-path classifier | 24 | Tier classification, dimension scoring, confidence calibration, performance |
| Semantic classifier | 15+ | Mocked logic, graceful degradation, config defaults, integration (real model) |
| Decision engine | 10 | Tier routing, degradation, fallback chains, cost estimation |
| Request dedup | 12 | Canonicalization, TTL, LRU eviction, stats |
| Session pinning | 16 | Fingerprinting, pin/retrieve, TTL, eviction, stats |
| Context compression | 11 | Whitespace, dedup, JSON compaction, thresholds |
| Configuration | 6 | YAML loading, defaults, model lookup |
| Integration (live) | 10 | vLLM provider, Gemini provider, full pipeline |
| **Total** | **104+** | |
