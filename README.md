<p align="center">
  <h1 align="center">Semantic Claw Router</h1>
  <p align="center">
    <em>System-Level Intelligent Router for LLM Mixture-of-Models</em>
  </p>
  <p align="center">
    Merging <a href="https://github.com/vllm-project/semantic-router">vLLM Semantic Router</a> architecture with <a href="https://github.com/BlockRunAI/ClawRouter">ClawRouter</a> intelligence
  </p>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <a href="#"><img src="https://img.shields.io/badge/tests-80%20passed-brightgreen.svg" alt="Tests"></a>
  <a href="#"><img src="https://img.shields.io/badge/routing-<1ms%20fast--path-orange.svg" alt="Fast Path"></a>
  <a href="https://github.com/vllm-project/semantic-router"><img src="https://img.shields.io/badge/lineage-vLLM%20Semantic%20Router-blueviolet.svg" alt="Lineage"></a>
  <a href="https://github.com/BlockRunAI/ClawRouter"><img src="https://img.shields.io/badge/influenced%20by-ClawRouter-green.svg" alt="Influenced By"></a>
</p>

---

## What Is This?

**Semantic Claw Router** is an intelligent LLM request routing layer that sits between client applications and multiple model backends. It brings together two complementary open-source projects:

| | [vLLM Semantic Router](https://github.com/vllm-project/semantic-router) | [ClawRouter](https://github.com/BlockRunAI/ClawRouter) |
|---|---|---|
| **Role** | Primary architecture & pipeline design | Influenced key routing intelligence |
| **Strengths** | Envoy ExtProc, signal-based decisions, 12+ selection algorithms, security pipeline, K8s CRDs | Fast-path weighted classifier, session pinning, request dedup, context compression, graceful degradation |
| **Stack** | Go / Rust / Python | TypeScript / Node.js |

This project implements the combined architecture in **Python**, validated against live model backends (vLLM on OpenShift + Google Gemini).

> **Why merge them?** Semantic Router provides the enterprise-grade pipeline architecture (Envoy ExtProc, boolean expression trees, neural classifiers, Kubernetes CRDs). ClawRouter contributes brilliant developer-experience innovations (sub-millisecond heuristic classification, session consistency, deduplication). Together, they form a router that is both production-grade *and* developer-friendly.

---

## Table of Contents

- [Key Features](#key-features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Pipeline Deep Dive](#pipeline-deep-dive)
- [Testing](#testing)
- [Performance](#performance)
- [Project Structure](#project-structure)
- [Feature Lineage](#feature-lineage)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License & Attribution](#license--attribution)

---

## Key Features

### Intelligent Routing

- **Fast-path pre-classifier** — A 15-dimension weighted scorer handles 70–80% of requests in **< 1 ms**, skipping expensive neural inference for obvious cases. Inspired by [ClawRouter's 14-dimension scorer](https://github.com/BlockRunAI/ClawRouter).
- **Complexity tiering** — Requests are classified into `SIMPLE`, `MEDIUM`, `COMPLEX`, or `REASONING` tiers, each mapped to the optimal model backend.
- **Agentic task detection** — Automatically identifies tool-use workflows, multi-step autonomous tasks, and iterative debugging patterns.
- **Confidence gating** — When the fast-path classifier isn't confident (sigmoid calibration below threshold), the request falls through to a full classification stage.

### Cost Optimization

- **Tier-based model mapping** — Route simple questions to free self-hosted models and complex reasoning to frontier APIs, saving 70–90% on inference costs.
- **Request deduplication** — SHA-256 content hashing with time-windowed LRU cache prevents duplicate inference on client retries.
- **Context auto-compression** — Whitespace normalization, paragraph deduplication, and JSON compaction reduce token usage on large contexts (>180 KB).

### Session Intelligence

- **Session pinning** — Multi-turn conversations are fingerprinted and pinned to the initially selected model, preventing jarring mid-conversation switches between different model personalities.
- **Graceful degradation** — When a provider returns errors, rate limits, or timeouts, the router automatically falls back to cheaper models rather than failing the request.
- **Context-aware fallback chains** — Fallback models are filtered by context window capacity (with 10% buffer) so the router never sends a 50K-token request to a 32K-context model.

### Multi-Provider Support

- **vLLM / OpenAI-compatible** — Routes to any backend speaking the OpenAI Chat Completions API (vLLM, Ollama, TGI, OpenAI, Azure).
- **Google Gemini** — Full bidirectional format translation (OpenAI ↔ Gemini API), including system instruction mapping and thinking-part filtering.
- **Extensible provider interface** — Add new providers by implementing `chat_completion()` and `health_check()`.

### Observability

- **Routing decision headers** — Every response includes `x-scr-model`, `x-scr-tier`, `x-scr-source`, `x-scr-latency-ms` for full transparency.
- **Prometheus-compatible metrics** — Request counts, latency percentiles (p50/p99), tier/model/source distributions, token usage, cost tracking.
- **Structured logging** — JSON or text format with configurable log levels.

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
│  Config: YAML (hot-reload planned)          │
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

For the full architecture specification (24 sections, 1,100+ lines), see **[architecture.md](./architecture.md)**.

---

## Quick Start

### Prerequisites

- Python 3.10+
- At least one LLM backend (vLLM, Ollama, OpenAI API, Gemini API, etc.)

### Installation

```bash
# Clone the repository
git clone https://github.com/cnuland/semantic-claw-router.git
cd semantic-claw-router

# Install in development mode
pip install -e ".[dev]"
```

### Configure

Create a configuration file with your model backends:

```yaml
# config.yaml
models:
  - name: "local-model"
    provider: "vllm"
    endpoint: "http://localhost:8000"     # Your vLLM / Ollama endpoint
    context_window: 32768
    cost_per_million_input: 0.0           # Self-hosted = free

  - name: "cloud-model"
    provider: "gemini"                    # or "vllm" for OpenAI-compatible
    endpoint: "https://generativelanguage.googleapis.com/v1beta"
    api_key: "${GEMINI_API_KEY}"          # From environment variable
    context_window: 1048576
    cost_per_million_input: 0.15
    cost_per_million_output: 0.60

# Route by complexity tier
default_tier_models:
  SIMPLE: "local-model"                  # Free, fast
  MEDIUM: "local-model"
  COMPLEX: "cloud-model"                 # Capable, paid
  REASONING: "cloud-model"

# Fallback when cloud provider fails
degradation:
  enabled: true
  fallback_model: "local-model"
```

### Run

```bash
# Set any required API keys
export GEMINI_API_KEY="your-api-key-here"

# Start the router
semantic-claw-router --config config.yaml --port 8080
```

### Send Requests

The router exposes an **OpenAI-compatible API** — point any OpenAI SDK client at it:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="unused",  # Router handles auth per-backend
)

# Simple question → routes to local model (free)
response = client.chat.completions.create(
    model="auto",  # Let the router decide
    messages=[{"role": "user", "content": "What is a Python decorator?"}],
)
print(response.choices[0].message.content)

# Complex reasoning → routes to cloud model (capable)
response = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": (
        "Prove by induction that the sum 1+2+...+n = n(n+1)/2. "
        "Show base case, inductive hypothesis, and inductive step."
    )}],
)
print(response.choices[0].message.content)
```

Or with `curl`:

```bash
# Simple request → local model
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "What is Python?"}]
  }'

# Check which model was selected (response headers)
# x-scr-model: local-model
# x-scr-tier: SIMPLE
# x-scr-source: fast_path
# x-scr-latency-ms: 42
```

### Using with OpenClaw

[OpenClaw](https://github.com/openclaw) is an open-source AI coding agent (similar to Claude Code). Since semantic-claw-router exposes an OpenAI-compatible API, OpenClaw can use it as a custom provider. Add this to `~/.openclaw/openclaw.json`:

```json5
{
  models: {
    providers: {
      "semantic-router": {
        baseUrl: "http://localhost:8080/v1",
        apiKey: "unused",  // Router handles per-backend auth
        api: "openai-completions",
        models: [
          {
            id: "auto",
            name: "Semantic Router (auto-select)",
            contextWindow: 1048576,
            maxTokens: 8192
          }
        ]
      }
    }
  }
}
```

Then select the model in OpenClaw: `/model semantic-router/auto`

OpenClaw's requests will flow through the router, which classifies each one:
- **Simple questions** (definitions, syntax, lookups) → free local model
- **Code generation** (functions, tests, reviews) → free local model
- **Complex reasoning** (proofs, system design, formal analysis) → cloud frontier model

In our benchmarks, **90% of typical coding assistant requests route to the free local model**, with only ~10% of requests (complex reasoning) hitting the paid cloud API.

> **Note**: This is the same integration pattern as [ClawRouter](https://github.com/BlockRunAI/ClawRouter), which was originally built for OpenClaw. The `baseUrl` approach also works with any OpenAI SDK client — LM Studio, Continue, Cursor, or any tool that supports custom API endpoints.
```

---

## Configuration

The router is configured via YAML. Full reference:

### Models

```yaml
models:
  - name: "qwen3-coder"           # Unique identifier
    provider: "vllm"              # "vllm" (OpenAI-compat) or "gemini"
    endpoint: "http://host:8000"  # Base URL
    api_key: "${API_KEY}"         # Optional, from env var
    context_window: 32768         # Max tokens
    cost_per_million_input: 0.0   # For cost tracking
    cost_per_million_output: 0.0
    supports_tools: true          # Tool/function calling
    supports_streaming: true      # SSE streaming
```

### Fast-Path Classifier

The 15-dimension weighted scorer that classifies most requests in < 1 ms:

```yaml
fast_path:
  enabled: true
  confidence_threshold: 0.7       # Below this → full classification
  weights:
    reasoning_markers: 0.18       # "prove", "derive", "induction"
    code_presence: 0.15           # Code blocks, function definitions
    multi_step_patterns: 0.12     # "first...then...finally"
    technical_terms: 0.10         # Domain-specific vocabulary
    token_count: 0.08             # Short vs. long inputs
    simple_indicators: 0.08       # "what is", "how do I"
    creative_markers: 0.05        # "write a story", "compose"
    question_complexity: 0.05     # Nested questions, conditionals
    constraint_indicators: 0.04   # "must", "ensure", "require"
    agentic_task: 0.04            # Tool-use, multi-step automation
    imperative_verbs: 0.03        # "implement", "design", "build"
    output_format: 0.03           # "return as JSON", "as a table"
    reference_complexity: 0.02    # Cross-references, citations
    domain_specificity: 0.02      # Medical, legal, scientific
    negation_complexity: 0.01     # Double negation, constraints
  tier_boundaries:
    simple: 0.0                   # Score < 0.3
    medium: 0.3                   # Score 0.3–0.5
    complex: 0.5                  # Score > 0.5 (+ reasoning override)
```

### Pipeline Stages

```yaml
# Request deduplication
dedup:
  enabled: true
  window_seconds: 30              # Cache TTL
  max_entries: 10000              # LRU capacity

# Session pinning (multi-turn consistency)
session:
  enabled: true
  ttl_seconds: 3600               # 1 hour session lifetime
  max_sessions: 10000

# Context auto-compression
compression:
  enabled: true
  threshold_bytes: 184320         # 180 KB trigger
  strategies:
    - whitespace                  # Collapse runs of whitespace
    - dedup                       # Remove duplicate paragraphs
    - json_compact                # Minify JSON in code fences

# Graceful degradation
degradation:
  enabled: true
  fallback_model: "local-model"
  triggers:
    - provider_error              # 5xx responses
    - rate_limit                  # 429 responses
    - timeout                    # Request timeout

# Observability
observability:
  log_level: "INFO"               # DEBUG, INFO, WARNING, ERROR
  log_format: "json"              # "json" or "text"
  metrics_enabled: true
```

See [examples/config.yaml](./examples/config.yaml) for a complete example.

---

## API Reference

The router exposes an **OpenAI-compatible HTTP API**:

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Route a chat completion request |
| `GET` | `/v1/models` | List configured model backends |
| `GET` | `/health` | Health check |
| `GET` | `/metrics` | Routing metrics summary (JSON) |

### Request Format

Standard [OpenAI Chat Completions](https://platform.openai.com/docs/api-reference/chat) format:

```json
{
  "model": "auto",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quicksort."}
  ],
  "max_tokens": 500,
  "temperature": 0.7
}
```

Set `"model": "auto"` to let the router classify and select. Or specify a model name directly to bypass classification.

### Response Headers

Every routed response includes these headers for observability:

| Header | Example | Description |
|--------|---------|-------------|
| `x-scr-model` | `qwen3-coder` | Which model handled the request |
| `x-scr-tier` | `SIMPLE` | Classified complexity tier |
| `x-scr-source` | `fast_path` | How the routing decision was made |
| `x-scr-latency-ms` | `42` | Total routing + inference latency |
| `x-scr-request-id` | `a1b2c3...` | Unique request identifier |
| `x-scr-dominant-signal` | `reasoning_markers` | Highest-weighted classification signal |
| `x-scr-dedup` | `true` | Whether response was served from dedup cache |
| `x-scr-degraded` | `true` | Whether a fallback model was used |

### Session Pinning

Pass an `x-session-id` header to pin multi-turn conversations:

```bash
# Turn 1 — router classifies and selects a model
curl -H "x-session-id: my-session" \
  http://localhost:8080/v1/chat/completions \
  -d '{"model":"auto","messages":[{"role":"user","content":"Hello"}]}'
# x-scr-source: fast_path

# Turn 2 — pinned to the same model automatically
curl -H "x-session-id: my-session" \
  http://localhost:8080/v1/chat/completions \
  -d '{"model":"auto","messages":[
    {"role":"user","content":"Hello"},
    {"role":"assistant","content":"Hi! How can I help?"},
    {"role":"user","content":"Write a sort function"}
  ]}'
# x-scr-source: session_pin
```

---

## Pipeline Deep Dive

### The 9-Stage Request Pipeline

```
Request In ──▶ ① Parse ──▶ ② Dedup ──▶ ③ Session ──▶ ④ Classify
                                                          │
              ⑨ Post-process ◀── ⑧ Fallback ◀── ⑦ Route ◀── ⑥ Compress ◀── ⑤ Decide
```

| Stage | What Happens | From |
|-------|-------------|------|
| **① Parse** | Extract messages, tools, model, max_tokens from OpenAI format | Semantic Router |
| **② Dedup** | SHA-256 hash after canonicalization; return cached if hit | ClawRouter |
| **③ Session** | Fingerprint conversation; check for existing model pin | ClawRouter |
| **④ Classify** | 15-dimension fast-path scorer (< 1 ms); fallback to full classify | ClawRouter |
| **⑤ Decide** | Map tier → model, estimate cost, check context window fit | Semantic Router |
| **⑥ Compress** | If > 180 KB: whitespace, dedup, JSON compaction | ClawRouter |
| **⑦ Route** | Forward to selected provider (vLLM or Gemini) | Semantic Router |
| **⑧ Fallback** | On error/429/timeout: try fallback chain, then degradation model | ClawRouter |
| **⑨ Post-process** | Update session pin, dedup cache, metrics; add response headers | Both |

### Fast-Path Classifier: The 15 Dimensions

The fast-path is the core innovation from ClawRouter — a regex-based scorer that avoids neural inference for clear-cut requests:

```
Input: "What is a Python decorator?"

  Dimension              Score    Weight   Contribution
  ─────────────────────  ──────   ──────   ────────────
  reasoning_markers       0.00    × 0.18   =  0.000
  code_presence           0.00    × 0.15   =  0.000
  multi_step_patterns     0.00    × 0.12   =  0.000
  technical_terms         0.20    × 0.10   =  0.020
  token_count            -0.50    × 0.08   = -0.040   ◀ Short input
  simple_indicators      -1.00    × 0.08   = -0.080   ◀ "What is" pattern
  ...
  ─────────────────────────────────────────────────────
  Weighted sum:                             = -0.067
  Nearest boundary:                           0.0 (SIMPLE/MEDIUM)
  Distance:                                   0.067
  Sigmoid confidence:                         0.76   ◀ Above 0.7 threshold
  Result:                                     SIMPLE ✓
```

When the weighted sum falls near a tier boundary, the sigmoid confidence drops below the threshold and the request is escalated to a full neural classifier (in production) or a conservative fallback (in this POC).

---

## Testing

### Run Unit Tests

```bash
# All unit tests (80 tests, < 1 second)
pytest tests/

# With coverage
pytest tests/ --cov=semantic_claw_router --cov-report=term-missing

# Specific test module
pytest tests/test_fastpath.py -v
```

### Run Integration Tests

Integration tests require live model backends:

```bash
# Set environment variables
export VLLM_ENDPOINT="https://your-vllm-endpoint"
export GEMINI_API_KEY="your-gemini-api-key"

# Run integration tests
pytest tests/test_integration.py -m integration -v
```

### Run the Example Runner

The example runner sends 21 simulated AI coding assistant prompts across all complexity tiers:

```bash
# Configure your endpoints in examples/config.yaml first
python examples/run_example.py
```

Output:

```
══════════════════════════════════════════════════════════════════
  Semantic Claw Router — Example Runner
══════════════════════════════════════════════════════════════════

  [ 1/21] Definition lookup
       Expected: SIMPLE
       Actual:   SIMPLE → local-model (fast_path)
       Latency:  42ms  Dedup: false  Agentic: false
       Response: "A Python decorator is a function that ..."
       Result:   ✓

  [ 2/21] Algorithm proof
       Expected: REASONING
       Actual:   REASONING → cloud-model (fast_path)
       Latency:  1203ms  Dedup: false  Agentic: false
       Response: "We prove by strong induction on ..."
       Result:   ✓
  ...

  Classification accuracy: 18/21 (86%)
```

### Test Coverage

| Module | Tests | Coverage |
|--------|-------|----------|
| Fast-path classifier | 24 | Tier classification, dimension scoring, confidence calibration, performance |
| Decision engine | 10 | Tier routing, degradation, fallback chains, cost estimation |
| Request dedup | 12 | Canonicalization, TTL, LRU eviction, stats |
| Session pinning | 16 | Fingerprinting, pin/retrieve, TTL, eviction, stats |
| Context compression | 11 | Whitespace, dedup, JSON compaction, thresholds |
| Configuration | 6 | YAML loading, defaults, model lookup |
| Integration (live) | 10 | vLLM provider, Gemini provider, full pipeline |
| **Total** | **89** | |

---

## Performance

### Routing Overhead

Benchmarked on Apple Silicon, Python 3.11:

| Metric | Target | Measured |
|--------|--------|----------|
| Fast-path classifier (per request) | < 1 ms | **27–46 μs** (0.03–0.05 ms) ✅ |
| Full pipeline overhead (no network) | < 5 ms | **34 μs** (0.034 ms) ✅ |
| End-to-end overhead (with live models) | < 5 ms | **0.1 ms** ✅ |
| Dedup cache lookup | < 0.1 ms | **< 0.01 ms** ✅ |
| Session pin lookup | < 0.1 ms | **< 0.01 ms** ✅ |

**The router adds 0.1 ms of overhead to each request.** Against a typical 200–800 ms inference call, that's **0.05% overhead** — essentially invisible.

### Model Usage (Cost Impact)

Against a representative 10-request coding assistant workload (definitions, code generation, debugging, proofs):

```
  qwen3-coder-next (local)     9/10 ( 90%)  ██████████████████  FREE
  gemini-2.5-flash (cloud)     1/10 ( 10%)  ██                  PAID

  → 90% of requests routed to the FREE self-hosted model
  → Only complex reasoning (proofs, formal analysis) hits the paid API
```

### Classification Accuracy

Against 21 diverse prompts (definition lookups, code generation, system design, formal proofs, agentic tool-use):

- **86% exact tier match** with the heuristic fast-path classifier alone
- Borderline cases (score near 0.3 MEDIUM/COMPLEX boundary) account for most misses
- Production deployments would add a neural BERT classifier as a second stage for ambiguous cases

---

## Project Structure

```
semantic-claw-router/
├── src/semantic_claw_router/
│   ├── __init__.py
│   ├── cli.py                    # CLI entrypoint
│   ├── config.py                 # YAML config loader + dataclasses
│   ├── server.py                 # HTTP server + 9-stage pipeline
│   ├── router/
│   │   ├── types.py              # Core types (ComplexityTier, ModelBackend, ...)
│   │   ├── fastpath.py           # 15-dimension fast-path classifier
│   │   └── decision.py           # Tier → model mapping + degradation
│   ├── pipeline/
│   │   ├── dedup.py              # Request deduplication (SHA-256 + LRU)
│   │   ├── session.py            # Session pinning (fingerprint + TTL)
│   │   └── compress.py           # Context auto-compression
│   ├── providers/
│   │   ├── base.py               # Abstract LLM provider interface
│   │   ├── vllm.py               # vLLM / OpenAI-compatible provider
│   │   └── gemini.py             # Google Gemini provider (format translation)
│   └── observability/
│       ├── logging.py            # Structured logging
│       └── metrics.py            # Prometheus metrics collection
├── tests/
│   ├── test_fastpath.py          # 24 classifier tests
│   ├── test_decision.py          # 10 decision engine tests
│   ├── test_dedup.py             # 12 deduplication tests
│   ├── test_session.py           # 16 session pinning tests
│   ├── test_compress.py          # 11 compression tests
│   ├── test_config.py            # 6 configuration tests
│   └── test_integration.py       # 10 live endpoint tests
├── examples/
│   ├── config.yaml               # Example configuration
│   └── run_example.py            # 21-prompt example runner
├── architecture.md               # Full architecture specification
├── research.md                   # Foundation projects analysis
├── skills.md                     # Capabilities reference
├── pyproject.toml                # Project metadata & dependencies
└── CLAUDE.md                     # Claude Code instructions
```

---

## Feature Lineage

This project merges two open-source LLM router projects. Every feature traces back to its origin. The **architecture and infrastructure** come from [vLLM Semantic Router](https://github.com/vllm-project/semantic-router). The **routing intelligence and developer-experience innovations** come from [ClawRouter](https://github.com/BlockRunAI/ClawRouter).

### From vLLM Semantic Router — The Skeleton

The structural patterns, infrastructure design, and production architecture:

| Feature | What It Does | Code |
|---------|-------------|------|
| **Pipeline architecture** | 9-stage sequential request processing (parse → classify → route → respond) | `server.py` |
| **OpenAI-compatible API** | Drop-in `/v1/chat/completions` — no client code changes needed | `server.py` |
| **Multi-provider abstraction** | Abstract `LLMProvider` interface; add new backends without touching core routing | `providers/base.py` |
| **Signal-based classification** | Each scoring dimension is an independent signal, evaluated in parallel | `router/fastpath.py` |
| **Decision engine** | Maps classification → model selection with cost estimation and fallback chains | `router/decision.py` |
| **Provider format translation** | Bidirectional OpenAI ↔ Gemini translation (messages, system instructions, finish reasons) | `providers/gemini.py` |
| **Context window capacity checks** | Fallback models filtered by context window (10% buffer) — never sends 50K tokens to a 32K model | `router/decision.py` |
| **Cost estimation & tracking** | Per-model pricing, per-request cost estimation, aggregate cost metrics | `decision.py`, `metrics.py` |
| **Observability system** | Prometheus-compatible metrics (p50/p99 latency), tier/model distributions, structured logging | `observability/` |
| **Response transparency headers** | `x-scr-model`, `x-scr-tier`, `x-scr-source`, `x-scr-latency-ms` on every response | `server.py` |
| **YAML configuration model** | Dataclass-based config with env var substitution, designed for hot-reload | `config.py` |
| **CLI entrypoint** | `semantic-claw-router --config config.yaml --port 8080` | `cli.py` |

### From ClawRouter — The Brain

The routing intelligence, developer experience, and cost-saving innovations:

| Feature | What It Does | Code |
|---------|-------------|------|
| **15-dimension fast-path classifier** | Regex-based weighted scorer handles 70–80% of requests in < 1 ms, skipping neural inference | `router/fastpath.py` |
| **Sigmoid confidence calibration** | Score near a tier boundary → low confidence → falls through to full classification | `router/fastpath.py` |
| **4-tier complexity model** | SIMPLE / MEDIUM / COMPLEX / REASONING with configurable score boundaries | `router/types.py` |
| **Tier-to-model mapping** | Direct config: `SIMPLE → local`, `REASONING → cloud` — simple, explicit, predictable | `config.py` |
| **Session pinning** | SHA-256 fingerprint of first user message; multi-turn conversations pinned to same model | `pipeline/session.py` |
| **Request deduplication** | JSON canonicalization → SHA-256 → LRU cache with 30s TTL prevents duplicate inference | `pipeline/dedup.py` |
| **Context auto-compression** | 3-strategy pipeline (whitespace, paragraph dedup, JSON compaction) triggers at 180 KB | `pipeline/compress.py` |
| **Agentic task detection** | Detects `tools[]` arrays, file/shell ops, iterative debugging patterns ("try again", "fix and re-run") | `router/fastpath.py` |
| **Graceful degradation** | On provider error/429/timeout: fallback chain sorted by cost, then degradation model | `router/decision.py` |
| **Dominant signal tracking** | Identifies which scoring dimension most influenced the routing decision | `router/fastpath.py` |
| **Reasoning keyword override** | 2+ reasoning keywords → force REASONING tier at 85% confidence, regardless of overall score | `router/fastpath.py` |

### What We Didn't Take from ClawRouter

| Feature | Why Not |
|---------|---------|
| **x402 blockchain payments** | Too niche for core infrastructure middleware |
| **Hardcoded 41-model catalog** | Dynamic YAML configuration is superior |
| **OpenClaw plugin interface** | Too tightly coupled to one client |

### Combined Feature Map

Every implemented feature in this project, traced to its origin:

| Feature | Origin | Module | Status |
|---------|--------|--------|--------|
| Pipeline architecture | vLLM Semantic Router | `server.py` | ✅ |
| OpenAI-compatible API | vLLM Semantic Router | `server.py` | ✅ |
| Multi-provider abstraction | vLLM Semantic Router | `providers/` | ✅ |
| OpenAI ↔ Gemini translation | vLLM Semantic Router | `providers/gemini.py` | ✅ |
| Decision engine | vLLM Semantic Router | `router/decision.py` | ✅ |
| Cost estimation | vLLM Semantic Router | `router/decision.py` | ✅ |
| Context window filtering | vLLM Semantic Router | `router/decision.py` | ✅ |
| Observability / metrics | vLLM Semantic Router | `observability/` | ✅ |
| Response headers | vLLM Semantic Router | `server.py` | ✅ |
| YAML config system | vLLM Semantic Router | `config.py` | ✅ |
| CLI entrypoint | vLLM Semantic Router | `cli.py` | ✅ |
| 15-dim fast-path classifier | ClawRouter | `router/fastpath.py` | ✅ |
| Sigmoid confidence gating | ClawRouter | `router/fastpath.py` | ✅ |
| 4-tier complexity model | ClawRouter | `router/types.py` | ✅ |
| Tier-to-model mapping | ClawRouter | `config.py` | ✅ |
| Session pinning | ClawRouter | `pipeline/session.py` | ✅ |
| Request deduplication | ClawRouter | `pipeline/dedup.py` | ✅ |
| Context auto-compression | ClawRouter | `pipeline/compress.py` | ✅ |
| Agentic task detection | ClawRouter | `router/fastpath.py` | ✅ |
| Graceful degradation | ClawRouter | `router/decision.py` | ✅ |
| Dominant signal tracking | ClawRouter | `router/fastpath.py` | ✅ |
| Reasoning keyword override | ClawRouter | `router/fastpath.py` | ✅ |
| Neural BERT classifiers | vLLM Semantic Router | — | 🔜 Planned |
| Boolean expression trees | vLLM Semantic Router | — | 🔜 Planned |
| Semantic caching (HNSW) | vLLM Semantic Router | — | 🔜 Planned |
| Security pipeline (PII, jailbreak) | vLLM Semantic Router | — | 🔜 Planned |
| Kubernetes CRDs + operator | vLLM Semantic Router | — | 🔜 Planned |
| Envoy ExtProc integration | vLLM Semantic Router | — | 🔜 Planned |
| RL-driven model selection | vLLM Semantic Router | — | 🔜 Planned |
| Cross-session agentic memory | vLLM Semantic Router | — | 🔜 Planned |

---

## Documentation

| Document | Description |
|----------|-------------|
| **[architecture.md](./architecture.md)** | Full architecture specification — 24 sections covering every pipeline stage, data flow, and component design |
| **[research.md](./research.md)** | Deep comparative analysis of vLLM Semantic Router and ClawRouter — what each project does, how they compare, and what we adopted from each |
| **[skills.md](./skills.md)** | Complete capabilities reference — 10 categories, 40+ skills with configuration examples |
| **[CLAUDE.md](./CLAUDE.md)** | Claude Code development instructions and project constraints |

---

## Contributing

We welcome contributions! The project is designed for easy extension:

### Development Setup

```bash
# Clone and install
git clone https://github.com/cnuland/semantic-claw-router.git
cd semantic-claw-router
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linter
ruff check src/ tests/

# Run example
python examples/run_example.py
```

### Areas for Contribution

- **New providers** — Add support for Anthropic, Azure, Bedrock, Ollama, etc.
- **New classifier dimensions** — Add scoring dimensions to the fast-path classifier
- **Neural classifiers** — Implement BERT-based classification for borderline cases
- **Semantic caching** — HNSW vector similarity cache for near-duplicate requests
- **Security pipeline** — Jailbreak detection, PII redaction
- **Kubernetes operator** — CRD-based configuration with `IntelligentPool` and `IntelligentRoute` resources
- **Streaming support** — True SSE streaming for Gemini provider (currently buffers)

---

## License & Attribution

This project is licensed under the **Apache License 2.0** — see [LICENSE](LICENSE) for details.

### Lineage

**Primary architecture**: [vllm-project/semantic-router](https://github.com/vllm-project/semantic-router) (Apache 2.0)
— Enterprise-grade intelligent LLM router with Envoy ExtProc, neural classifiers, and Kubernetes integration.

**Influenced by**: [BlockRunAI/ClawRouter](https://github.com/BlockRunAI/ClawRouter) (MIT License)
— Developer-friendly model router with fast-path weighted classification, session pinning, and request deduplication.

The following concepts were re-implemented from ClawRouter's design (not copied code):
- Tiered complexity classification with confidence gating
- Session pinning for multi-turn conversation consistency
- Request deduplication via content hashing
- Context auto-compression strategies
- Agentic task auto-detection
- Graceful degradation with fallback chains
