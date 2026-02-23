<p align="center">
  <h1 align="center">Semantic Claw Router</h1>
  <p align="center">
    <em>Intelligent LLM Request Router — Mixture-of-Models at the System Level</em>
  </p>
  <p align="center">
    Merging <a href="https://github.com/vllm-project/semantic-router">vLLM Semantic Router</a> architecture with <a href="https://github.com/BlockRunAI/ClawRouter">ClawRouter</a> intelligence
  </p>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <a href="#"><img src="https://img.shields.io/badge/tests-89%20passed-brightgreen.svg" alt="Tests"></a>
  <a href="#"><img src="https://img.shields.io/badge/routing-<1ms%20fast--path-orange.svg" alt="Fast Path"></a>
  <a href="https://github.com/vllm-project/semantic-router"><img src="https://img.shields.io/badge/lineage-vLLM%20Semantic%20Router-blueviolet.svg" alt="Lineage"></a>
  <a href="https://github.com/BlockRunAI/ClawRouter"><img src="https://img.shields.io/badge/influenced%20by-ClawRouter-green.svg" alt="Influenced By"></a>
</p>

---

## What Is This?

**Semantic Claw Router** is an intelligent LLM request routing layer that sits between client applications and multiple model backends. It classifies each request by complexity and routes it to the optimal model — saving cost while preserving quality.

| | [vLLM Semantic Router](https://github.com/vllm-project/semantic-router) | [ClawRouter](https://github.com/BlockRunAI/ClawRouter) |
|---|---|---|
| **Role** | Primary architecture & pipeline design | Routing intelligence & developer experience |
| **Strengths** | Pipeline architecture, decision engine, multi-provider abstraction, observability, K8s CRDs | Fast-path classifier, session pinning, request dedup, context compression, graceful degradation |
| **Stack** | Go / Rust / Python | TypeScript / Node.js |

This project implements the combined architecture in **Python**, validated against live model backends (vLLM on OpenShift + Google Gemini).

> **Why merge them?** Semantic Router provides the enterprise-grade pipeline architecture. ClawRouter contributes developer-experience innovations (sub-millisecond heuristic classification, session consistency, deduplication). Together: production-grade *and* developer-friendly.

---

## Key Results

```
  90% of requests → FREE self-hosted model (qwen3-coder)
  10% of requests → paid cloud API (gemini-2.5-flash)

  Routing overhead:  0.1 ms  (0.05% of inference time)
  Fast-path classifier:  27–46 μs per request
  Classification accuracy:  86% exact tier match
  Test suite:  89 tests (80 unit + 9 integration)
```

Only requests requiring genuine mathematical reasoning hit the paid cloud API. Everything else — definitions, code generation, debugging, documentation — routes to the free local model.

---

## Architecture

```
Client Applications (OpenAI-compatible API)
              │
              ▼
┌─────────────────────────────────────────────┐
│       SEMANTIC CLAW ROUTER                  │
│                                             │
│  ┌─── Request Pipeline (9 stages) ───────┐  │
│  │                                       │  │
│  │  1. Parse request                     │  │
│  │  2. Dedup check (SHA-256 LRU)         │  │
│  │  3. Session pin lookup                │  │
│  │  4. Fast-path classify (15-dim, <1ms) │  │
│  │  5. Decision engine (tier → model)    │  │
│  │  6. Context compression (>180KB)      │  │
│  │  7. Provider routing                  │  │
│  │  8. Fallback on error (chain)         │  │
│  │  9. Post-process (metrics, headers)   │  │
│  │                                       │  │
│  └───────────────────────────────────────┘  │
│                                             │
│  Config: YAML with ${ENV} expansion         │
│  Metrics: Prometheus-compatible             │
│  API: OpenAI Chat Completions               │
└──────────────┬──────────────────────────────┘
               │
               ▼
     Backend Model Pools
  ┌──────────┐  ┌──────────┐  ┌───────┐
  │  vLLM /  │  │  Google  │  │  Any  │
  │  OpenAI  │  │  Gemini  │  │ other │
  └──────────┘  └──────────┘  └───────┘
```

---

## Quick Start

```bash
# Install
git clone https://github.com/cnuland/semantic-claw-router.git
cd semantic-claw-router
pip install -e ".[dev]"

# Configure (edit examples/config.yaml with your endpoints)
export GEMINI_API_KEY="your-key"

# Run
semantic-claw-router --config examples/config.yaml --port 8080
```

Send a request:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "What is Python?"}]}'

# Response headers show routing decision:
# x-scr-model: local-model
# x-scr-tier: SIMPLE
# x-scr-source: fast_path
```

Or use any OpenAI SDK client:

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8080/v1", api_key="unused")
response = client.chat.completions.create(
    model="auto",  # Let the router decide
    messages=[{"role": "user", "content": "What is a Python decorator?"}],
)
```

For full setup instructions including OpenShift/Kubernetes credentials and OpenClaw integration, see **[docs/getting-started.md](docs/getting-started.md)**.

---

## Feature Lineage

Every feature traces back to its origin project.

### From vLLM Semantic Router — The Skeleton

| Feature | Code |
|---------|------|
| 9-stage sequential request pipeline | `server.py` |
| OpenAI-compatible `/v1/chat/completions` API | `server.py` |
| Abstract `LLMProvider` multi-provider interface | `providers/base.py` |
| Bidirectional OpenAI ↔ Gemini format translation | `providers/gemini.py` |
| Decision engine (tier → model + cost estimation) | `router/decision.py` |
| Context window capacity checks (10% buffer) | `router/decision.py` |
| Prometheus-compatible observability (p50/p99, distributions) | `observability/` |
| Response transparency headers (`x-scr-*`) | `server.py` |
| YAML config with env var expansion | `config.py` |
| CLI entrypoint | `cli.py` |

### From ClawRouter — The Brain

| Feature | Code |
|---------|------|
| 15-dimension fast-path classifier (< 1 ms) | `router/fastpath.py` |
| Sigmoid confidence calibration + gating | `router/fastpath.py` |
| 4-tier complexity model (SIMPLE/MEDIUM/COMPLEX/REASONING) | `router/types.py` |
| Session pinning (SHA-256 fingerprint, multi-turn consistency) | `pipeline/session.py` |
| Request deduplication (JSON canonicalization → SHA-256 → LRU) | `pipeline/dedup.py` |
| Context auto-compression (whitespace, dedup, JSON compaction) | `pipeline/compress.py` |
| Agentic task detection (tools[], file ops, iterative debugging) | `router/fastpath.py` |
| Graceful degradation (fallback chains on error/429/timeout) | `router/decision.py` |
| Reasoning keyword override (2+ keywords → force REASONING) | `router/fastpath.py` |
| Dominant signal tracking | `router/fastpath.py` |

---

## Documentation

| Document | Description |
|----------|-------------|
| **[Getting Started](docs/getting-started.md)** | Installation, configuration, running the router, OpenClaw integration |
| **[Configuration Reference](docs/configuration.md)** | Full YAML config — models, classifier weights, pipeline stages |
| **[API Reference](docs/api-reference.md)** | Endpoints, request format, response headers, session pinning |
| **[Pipeline Deep Dive](docs/pipeline.md)** | 9-stage pipeline walkthrough, 15-dimension classifier details |
| **[Benchmarks](docs/benchmarks.md)** | Detailed performance data, classifier timings, test coverage |
| **[Architecture](docs/architecture.md)** | Full system architecture specification |
| **[Research](docs/research.md)** | Comparative analysis of upstream projects |
| **[Contributing](docs/contributing.md)** | Development setup, project structure, areas for contribution |

---

## License & Attribution

Licensed under **Apache License 2.0** — see [LICENSE](LICENSE).

**Primary architecture**: [vllm-project/semantic-router](https://github.com/vllm-project/semantic-router) (Apache 2.0)

**Influenced by**: [BlockRunAI/ClawRouter](https://github.com/BlockRunAI/ClawRouter) (MIT License) — concepts re-implemented, not copied code.
