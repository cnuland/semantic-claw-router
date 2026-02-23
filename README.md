# Semantic Router

> **System-Level Intelligent Router for Mixture-of-Models at Cloud, Data Center, and Edge**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

An intelligent LLM request routing layer that sits between client applications and multiple model backends, providing semantic classification, cost-aware model selection, security enforcement, caching, and observability — all at wire speed.

**Influenced by** [ClawRouter](https://github.com/BlockRunAI/ClawRouter) by BlockRun (MIT License) — incorporating concepts of tiered complexity classification with confidence gating, session pinning, request deduplication, and context auto-compression.

---

## Why Semantic Router?

Modern AI applications face five critical challenges:

1. **Signal Capture** — How to extract the right signals from requests (domain, complexity, intent, modality) to make intelligent routing decisions.
2. **Signal Combination** — How to combine heterogeneous signals (keywords, embeddings, neural classifiers) into actionable routing decisions.
3. **Inter-Model Collaboration** — How to enable efficient collaboration between different models (escalation, parallel execution, re-evaluation).
4. **Security** — How to guard against jailbreaks, PII leaks, and hallucinations at the system level.
5. **Self-Learning** — How to build self-improving systems from collected signals (RL-driven selection, Elo ratings, feedback loops).

Semantic Router solves all five with a single, drop-in middleware layer.

---

## Key Features

### Intelligent Routing
- **Multi-signal classification**: Keywords, embeddings, BERT/ModernBERT neural classifiers, complexity assessment, language detection, modality detection — all running in parallel
- **Fast-path pre-classifier**: Lightweight 14-dimension weighted scorer handles 70-80% of requests in <1ms, skipping expensive neural inference for obvious cases
- **Boolean expression tree decisions**: Combine signals with AND/OR/NOT logic for precise routing rules
- **Agentic task detection**: Automatically identifies multi-step autonomous workflows and routes to agent-optimized models

### Model Selection
- **12+ selection algorithms**: Static, Elo, RouterDC, AutoMix, Hybrid, KNN, KMeans, SVM, MLP, RL-driven, GMTRouter, Latency-aware
- **Session pinning**: Multi-turn conversations stay on the same model for consistency
- **Graceful degradation**: When rate limits or budgets are exceeded, automatically fall back to cheaper models instead of failing
- **Context-aware fallback chains**: Fallback models filtered by context window capacity

### Security
- **Jailbreak detection**: PromptGuard-based adversarial input detection via native Rust/Candle inference
- **PII detection & redaction**: Token-level PII classification with configurable policies per routing decision
- **Hallucination mitigation**: Fact-checking and NLI-based response grounding
- **Rate limiting**: Envoy RLS integration with degradation policies

### Performance
- **Request deduplication**: SHA-256 content hashing with time-windowed LRU cache prevents duplicate inference on client retries
- **Semantic caching**: In-memory, Redis, or Milvus backends with SIMD-optimized similarity search
- **Context auto-compression**: Automatic whitespace normalization, deduplication, and JSON compaction for large contexts
- **Zero-downtime config reload**: Atomic pointer swap for hot configuration changes

### Infrastructure
- **Envoy ExtProc native**: Deploys as an Envoy external processor — no application code changes required
- **Kubernetes operator**: Full CRD-based configuration with `IntelligentPool` and `IntelligentRoute` resources
- **Multi-provider support**: vLLM, OpenAI, Anthropic, Azure, Bedrock, Gemini, Vertex AI, DeepSeek, and more
- **Full observability**: Prometheus metrics, OpenTelemetry/Jaeger/Zipkin tracing, structured logging

---

## Architecture Overview

```
Client Applications (OpenAI-compatible API)
              │
              ▼
         Envoy Proxy (TLS, load balancing)
              │ gRPC ExtProc
              ▼
┌─────────────────────────────────────────┐
│          SEMANTIC ROUTER CORE           │
│                                         │
│  Request Pipeline:                      │
│   1. Deduplication                      │
│   2. Fast-path classifier (<1ms)        │
│   3. Full signal extraction (parallel)  │
│   4. Decision engine (expr trees)       │
│   5. Security (jailbreak, PII, rates)   │
│   6. Cache lookup                       │
│   7. RAG injection                      │
│   8. Context compression                │
│   9. Model selection + session pinning  │
│                                         │
│  ML Inference: Rust/Candle (no Python)  │
│  Config: YAML + K8s CRDs (hot-reload)  │
└─────────────┬───────────────────────────┘
              │
              ▼
    Backend Model Pools
  (vLLM, OpenAI, Anthropic, Gemini, ...)
```

For the full architecture document, see [architecture.md](./architecture.md).

---

## Quick Start

### Prerequisites

- Go 1.22+
- Rust 1.75+ (for Candle ML bindings)
- Envoy Proxy 1.30+
- Docker (optional, for containerized deployment)

### Local Development

```bash
# Clone the repository
git clone https://github.com/YOUR_ORG/semantic-router.git
cd semantic-router

# Build the router (with Candle ML support)
cd src/semantic-router
make build-candle

# Copy and edit the configuration
cp config/config.example.yaml config/config.yaml
# Edit config.yaml with your model endpoints and routing decisions

# Start the router
./bin/semantic-router --config config/config.yaml

# Start Envoy with the ExtProc configuration
envoy -c deploy/envoy/envoy.yaml
```

### Docker

```bash
# Build the container image
docker build -t semantic-router:latest .

# Run with your configuration
docker run -p 8080:8080 -v $(pwd)/config:/config semantic-router:latest
```

### Kubernetes with Helm

```bash
# Add the Helm repository
helm repo add semantic-router https://YOUR_ORG.github.io/semantic-router/charts

# Install with default values
helm install sr semantic-router/semantic-router \
  --namespace semantic-router \
  --create-namespace \
  -f values.yaml
```

### Using the CLI

```bash
# Install the CLI tool
pip install vllm-sr

# Initialize a new configuration
vllm-sr init

# Start the router with the dashboard
vllm-sr serve --dashboard
```

---

## Configuration

The router is configured via YAML with hot-reload support. Key sections:

```yaml
# Routing decisions — the core of the configuration
decisions:
  - name: "code-generation"
    rules:
      - type: "AND"
        conditions:
          - signal: "domain"
            category: "coding"
          - signal: "complexity"
            level: "high"
    model: "deepseek-coder-v3"
    plugins:
      sessionPinning:
        enabled: true
      cache:
        enabled: true
        ttl: 300

# Fast-path classifier (influenced by ClawRouter)
fastPathClassifier:
  enabled: true
  confidenceThreshold: 0.7
  weights:
    codePresence: 0.15
    reasoningMarkers: 0.18
    multiStepPatterns: 0.12

# Request deduplication
deduplication:
  enabled: true
  windowSeconds: 30

# Backend model definitions
backendModels:
  - name: "gpt-4o"
    provider: "openai"
    pricing: { input: 2.50, output: 10.00 }
  - name: "claude-sonnet-4"
    provider: "anthropic"
    pricing: { input: 3.00, output: 15.00 }
```

See the [full configuration reference](docs/configuration.md) for all options.

---

## Project Structure

```
semantic-router/
├── src/
│   ├── semantic-router/          # Go — Core router engine
│   │   ├── cmd/                  #   CLI entrypoint
│   │   └── pkg/                  #   Library packages
│   │       ├── extproc/          #     Envoy ExtProc + filter pipeline
│   │       ├── decision/         #     Boolean expression tree engine
│   │       ├── classification/   #     Multi-signal classifiers
│   │       ├── selection/        #     Model selection algorithms
│   │       ├── fastpath/         #     Fast-path pre-classifier
│   │       ├── session/          #     Session pinning & tracking
│   │       ├── dedup/            #     Request deduplication
│   │       ├── compression/      #     Context auto-compression
│   │       ├── cache/            #     Semantic caching
│   │       ├── hnsw/             #     HNSW vector index (SIMD)
│   │       ├── memory/           #     Agentic cross-session memory
│   │       ├── looper/           #     Multi-model execution
│   │       ├── mcp/              #     Model Context Protocol client
│   │       ├── k8s/              #     Kubernetes CRD controller
│   │       └── observability/    #     Metrics, tracing, logging
│   ├── candle-binding/           # Rust — Native ML inference
│   ├── training/                 # Python — Model training pipelines
│   └── vllm-sr/                  # Python — CLI tool
├── deploy/
│   ├── helm/                     # Helm charts
│   ├── operator/                 # Kubernetes operator
│   ├── envoy/                    # Envoy configurations
│   └── manifests/                # Raw Kubernetes manifests
├── config/                       # Example configurations
├── docs/                         # Documentation
├── architecture.md               # Architecture deep-dive
├── research.md                   # Research foundations
└── skills.md                     # Capabilities reference
```

---

## Influenced By

This project incorporates architectural concepts from [ClawRouter](https://github.com/BlockRunAI/ClawRouter) by BlockRun, including:

- **Tiered complexity classification with confidence gating** — A multi-dimension weighted scorer that provides a fast pre-classification path, falling back to full neural classification only when confidence is low.
- **Session pinning** — Multi-turn conversations are pinned to the initially selected model to prevent jarring mid-conversation switches.
- **Request deduplication** — Content-hash-based dedup prevents duplicate inference on client retries.
- **Context auto-compression** — Large contexts are automatically compressed (whitespace normalization, dedup, JSON compaction) before routing.
- **Agentic task auto-detection** — Automatic identification of multi-step autonomous workflows for model routing optimization.
- **Graceful degradation** — Budget-exhaustion triggers automatic fallback to cheaper models rather than request failure.

ClawRouter is licensed under the MIT License. These concepts have been re-implemented in Go for the semantic-router ExtProc pipeline.

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Install Go dependencies
cd src/semantic-router && go mod download

# Install Rust toolchain (for Candle bindings)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Run tests
make test

# Run linter
make lint

# Build everything
make build-all
```

---

## License

This project is licensed under the Apache License 2.0 — see [LICENSE](LICENSE) for details.

Portions of this project are influenced by [ClawRouter](https://github.com/BlockRunAI/ClawRouter), which is licensed under the MIT License.
