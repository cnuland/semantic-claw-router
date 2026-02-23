# Research: Foundation Projects Analysis

> Comprehensive analysis of the two projects informing this router's design.
> Last updated: 2026-02-23

---

## Table of Contents

- [1. vllm-project/semantic-router](#1-vllm-projectsemantic-router)
  - [1.1 Overview](#11-overview)
  - [1.2 Architecture](#12-architecture)
  - [1.3 Request Routing Pipeline](#13-request-routing-pipeline)
  - [1.4 Signal Classification System](#14-signal-classification-system)
  - [1.5 Model Selection Algorithms](#15-model-selection-algorithms)
  - [1.6 Multi-Model Execution](#16-multi-model-execution)
  - [1.7 Security Pipeline](#17-security-pipeline)
  - [1.8 Configuration Model](#18-configuration-model)
  - [1.9 Key Design Patterns](#19-key-design-patterns)
  - [1.10 Tech Stack](#110-tech-stack)
  - [1.11 vLLM Integration](#111-vllm-integration)
  - [1.12 Extension Mechanisms](#112-extension-mechanisms)
- [2. BlockRunAI/ClawRouter](#2-blockrunaiclawrouter)
  - [2.1 Overview](#21-overview)
  - [2.2 Architecture](#22-architecture)
  - [2.3 Two-Stage Classification System](#23-two-stage-classification-system)
  - [2.4 The 14 Scoring Dimensions](#24-the-14-scoring-dimensions)
  - [2.5 Complexity Tiers](#25-complexity-tiers)
  - [2.6 Payment Innovation (x402)](#26-payment-innovation-x402)
  - [2.7 Key Design Patterns](#27-key-design-patterns)
  - [2.8 Tech Stack](#28-tech-stack)
  - [2.9 Unique Innovations](#29-unique-innovations)
- [3. Comparative Analysis](#3-comparative-analysis)
- [4. Synthesis: What We Take from Each](#4-synthesis-what-we-take-from-each)

---

## 1. vllm-project/semantic-router

### 1.1 Overview

The vLLM Semantic Router is a **system-level intelligent router for Mixture-of-Models** designed for cloud, data center, and edge deployments. It is production-grade middleware that sits between client applications and multiple LLM backends.

- **Repository**: https://github.com/vllm-project/semantic-router
- **License**: Apache 2.0
- **Stars**: 3,200+ | **Contributors**: 73+
- **Recognition**: NeurIPS 2025 MLForSys workshop paper, #1 on RouterArena (Feb 2026)
- **Current version**: v0.1 "Iris" (Jan 2026)

The project addresses five core challenges in LLM serving:
1. **Signal Capture** — Extracting meaningful signals from requests
2. **Signal Combination** — Combining heterogeneous signals for routing decisions
3. **Inter-Model Collaboration** — Enabling multi-model execution strategies
4. **Security** — System-level protection against attacks and data leaks
5. **Self-Learning** — Continuous improvement from collected data

### 1.2 Architecture

The system is organized into four main components:

#### A. Core Router (`src/semantic-router/`, Go)
The primary engine, implemented as an Envoy ExtProc gRPC service. This is the largest component (Go is 43.4% of the codebase). Key packages:

| Package | Purpose | File Count |
|---------|---------|------------|
| `extproc/` | Envoy External Processor filter chain | 48 files |
| `classification/` | Multi-signal classifiers | 34 files |
| `selection/` | Model selection algorithms | 23 files |
| `decision/` | Boolean expression tree evaluator | ~10 files |
| `cache/` | Semantic caching (memory/Redis/Milvus) | 15 files |
| `hnsw/` | HNSW vector index with SIMD | ~8 files |
| `looper/` | Multi-model execution strategies | 9 files |
| `memory/` | Agentic cross-session memory | 13 files |
| `mcp/` | Model Context Protocol client | ~6 files |
| `k8s/` | Kubernetes CRD controller | ~5 files |
| `config/` | Configuration model | ~4 files |
| `observability/` | Metrics, tracing, logging | ~6 files |

#### B. Candle Bindings (`candle-binding/`, Rust + Go)
Rust library using Hugging Face's Candle framework for native ML inference:
- BERT/ModernBERT/mmBERT-32K classification
- PromptGuard jailbreak detection
- PII token classification
- Embedding generation

This eliminates the need for a Python runtime in production.

#### C. Training Pipelines (`src/training/`, Python)
12 training pipelines for the ML models used by the router:
- BERT classifier fine-tuning
- PII model training
- PromptGuard fine-tuning
- Embedding model training (cache, domain-adapted)
- ML model selection training (KNN, KMeans, SVM, MLP)
- RL model selection training
- Multi-task BERT and LoRA adapter training

#### D. CLI Tool (`src/vllm-sr/`, Python)
The `vllm-sr` command-line tool (pip-installable):
- `vllm-sr init` — Initialize configuration
- `vllm-sr serve` — Start the router
- `vllm-sr config` — Manage configuration
- `vllm-sr dashboard` — Launch web dashboard
- `vllm-sr status/logs/stop` — Operational commands

### 1.3 Request Routing Pipeline

The routing pipeline operates as an Envoy ExtProc gRPC service with three processing phases:

#### Request Header Phase
1. Extract OpenTelemetry trace context
2. Start root tracing span
3. Store headers in RequestContext
4. Detect streaming mode
5. Route special endpoints (`/v1/models`, Response API)

#### Request Body Phase (Core Pipeline — 13 stages)
1. **Response API Translation** — Convert `/v1/responses` to chat completion format
2. **Stream Parameter Extraction** — Parse `stream=true` from body
3. **Request Parsing** — Parse OpenAI format, extract model name
4. **Looper Detection** — Check for internal multi-model sub-requests
5. **Decision Evaluation** — The core semantic routing step:
   - Run all signal extractors in parallel
   - Evaluate boolean expression trees over signal matches
   - Select the best matching decision by confidence/priority
6. **Jailbreak Detection** — Block adversarial inputs
7. **PII Detection** — Redact or block based on policy
8. **Rate Limiting** — Enforce quotas (429 responses)
9. **Cache Lookup** — Check semantic cache for similar requests
10. **RAG Plugin Execution** — Retrieve augmented context
11. **Modality Classification** — Route image generation to diffusion backends
12. **Memory Retrieval** — Inject cross-session agentic context
13. **Model Routing** — Three paths:
    - Anthropic routing (format transformation)
    - Auto model selection (decision engine + selection algorithms)
    - Direct model routing (explicit model requests)

#### Response Body Phase
1. Record TTFT/TPOT latency metrics
2. Accumulate streaming chunks
3. Report token usage to rate limiters
4. Compute cost metrics
5. Cache responses with decision-specific TTL
6. Run hallucination detection
7. Extract memories asynchronously

### 1.4 Signal Classification System

Signals are extracted in parallel by the `Classifier.EvaluateAllSignalsWithContext()` method:

| Signal Type | Method | Purpose |
|-------------|--------|---------|
| Keyword | Regex, BM25, n-gram matching | Pattern-based routing rules |
| Embedding | Qwen3/Gemma/BERT vector similarity | Semantic similarity matching |
| Domain | BERT/ModernBERT neural classification | Domain/category detection |
| Fact-check | Neural classifier | Detect fact-check necessity |
| Feedback | Neural classifier | User satisfaction detection |
| Language | Heuristic + neural | Input language identification |
| Complexity | Multi-factor assessment | Task complexity estimation |
| Modality | AR vs. Diffusion detection | Image generation routing |
| Authorization | Rule-based | Access control evaluation |
| Context | Window size analysis | Context length requirements |

All signals produce a `SignalMatches` struct that feeds into the Decision Engine.

### 1.5 Model Selection Algorithms

The router implements 12+ selection algorithms, all implementing a common `Selector` interface:

| Algorithm | Type | Description |
|-----------|------|-------------|
| Static | Rule-based | Configuration-based scoring |
| Elo | Statistical | Bradley-Terry rating with time decay |
| RouterDC | Neural | Dual-contrastive learning embeddings |
| AutoMix | Optimization | POMDP-based cost-quality optimization |
| Hybrid | Ensemble | Weighted combination of multiple methods |
| KNN | ML | K-nearest neighbors classification |
| KMeans | ML | Cluster-based selection |
| SVM | ML | Support vector machine classification |
| MLP | ML | Multi-layer perceptron classification |
| RL-Driven | RL | Thompson Sampling with personalization |
| GMTRouter | Graph | Graph-based personalized routing |
| Latency-Aware | Metrics | TPOT/TTFT percentile thresholds |

Algorithms are managed via a Factory + Registry pattern, enabling runtime switching.

### 1.6 Multi-Model Execution

The `looper/` package implements strategies for sending requests to multiple models:

| Strategy | Behavior |
|----------|----------|
| Confidence | Escalate through models until confidence threshold met |
| Ratings | Parallel execution with rating-based winner selection |
| ReMoM | Re-evaluation of Multiple Models (multi-round) |
| RL-Driven | Reinforcement learning-based sequencing |

### 1.7 Security Pipeline

Three security stages run sequentially in the request pipeline:

1. **Jailbreak Detection**: PromptGuard model via Candle (no Python). Configurable threshold per decision.
2. **PII Detection**: Token-level classification identifying names, emails, SSNs, etc. Configurable action (block, redact, log) per decision.
3. **Hallucination Mitigation**: Response-phase NLI-based fact-checking against source documents.

### 1.8 Configuration Model

Top-level `RouterConfig` struct with these major sections:

- **Decisions[]** — Routing rules (boolean expression trees, model refs, plugins)
- **Categories[]** — Domain categories for classification
- **Signals[]** — Signal extraction rules per type
- **BackendModels** — Model parameters, pricing, LoRA adapters
- **vLLMEndpoints** — vLLM server endpoints with load balancing
- **ProviderProfiles** — Cloud provider configs (OpenAI, Azure, Anthropic, Bedrock, Gemini, Vertex)
- **InlineModels** — Built-in ML models (embeddings, classifiers, guards)
- **ExternalModels** — LLM-based classifiers for specialized tasks
- **ModelSelection** — Algorithm configuration and feedback settings
- **SemanticCache** — Cache backend, similarity thresholds, TTL, HNSW params
- **ResponseAPI** — Stateful conversation storage
- **RouterReplay** — Decision recording storage
- **Memory** — Agentic memory configuration
- **AuthzConfig** — Credential resolution chains
- **RateLimitConfig** — Rate limiter configuration
- **Observability** — Metrics, tracing, logging settings

Configuration sources:
- YAML files with filesystem watcher (hot-reload)
- Kubernetes CRDs (`IntelligentPool`, `IntelligentRoute`)
- Environment variables

### 1.9 Key Design Patterns

1. **Envoy ExtProc** — Integration with Envoy proxy as a gRPC external processor
2. **Filter Chain / Pipeline** — Sequential filter stages (`req_filter_*.go`, `res_filter_*.go`)
3. **Signal-Based Decision Engine** — Parallel signal extraction → boolean expression tree evaluation
4. **Factory + Registry** — Runtime-switchable algorithm implementations
5. **Atomic Pointer Hot-Reload** — Zero-downtime configuration changes via `atomic.Pointer`
6. **Strategy Pattern** — Common `Selector` interface for all model selection algorithms
7. **Build Tag Conditional Compilation** — CGo features with stub fallbacks
8. **Graceful Degradation** — Failed optional operations log and continue
9. **Kubernetes Operator** — Kubebuilder-based CRD management
10. **Provider Chain** — Cascading credential and rate-limit providers

### 1.10 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Core Router | Go 1.22+ | Performance-critical request processing |
| ML Inference | Rust (Candle) | Native BERT/embedding inference without Python |
| Training | Python (PyTorch, HuggingFace) | ML model training pipelines |
| CLI | Python (Click, Pydantic) | Developer tooling |
| Dashboard | TypeScript (React) | Web UI for monitoring |
| Proxy | Envoy | Front-facing load balancer and TLS |
| Vector DB | Milvus | Semantic cache, memory, replay storage |
| Cache | Redis | Alternative cache/session backend |
| RDBMS | PostgreSQL | Alternative replay storage |
| Metrics | Prometheus | Time-series metrics |
| Tracing | OpenTelemetry / Jaeger / Zipkin | Distributed tracing |
| Orchestration | Kubernetes + Helm | Deployment and configuration |
| SIMD | AMD64 Assembly | Distance computation optimization |

### 1.11 vLLM Integration

- **Front proxy**: Sits before vLLM instances, routing via Envoy `ORIGINAL_DST` clusters
- **OpenAI-compatible API**: Seamless interception of standard chat completion requests
- **LoRA adapter routing**: Selects specific LoRA adapters served by vLLM
- **vLLM as classifier backend**: Uses vLLM instances for running classification models
- **Kubernetes CRDs**: `IntelligentPool` and `IntelligentRoute` under `apis/vllm.ai/v1alpha1`

### 1.12 Extension Mechanisms

1. **Decision Plugins** — Per-decision toggles for cache, jailbreak, PII, hallucination, memory, replay, system prompts, reasoning mode
2. **MCP Integration** — Model Context Protocol client (stdio + HTTP) for tool discovery and invocation
3. **Custom Selection Algorithms** — Implement `Selector` interface, register via factory
4. **Multiple RAG Backends** — Cache, external, hybrid, MCP, Milvus, OpenAI, vector store
5. **Pluggable Storage** — In-memory, Redis, Milvus, PostgreSQL backends
6. **Provider Profiles** — Add new LLM providers via configuration (no code changes)
7. **Kubernetes Operator** — CRD-based declarative management
8. **Build Tags** — Feature gating for native library dependencies
9. **Embedding Auto-Detection** — Supports Qwen3, Gemma, mmBERT, standard BERT
10. **Helm Chart** — Full deployment customization via values files

---

## 2. BlockRunAI/ClawRouter

### 2.1 Overview

ClawRouter is an **intelligent local LLM routing proxy** built by BlockRun. It sits between an AI coding assistant framework (OpenClaw) and multiple LLM API providers.

- **Repository**: https://github.com/BlockRunAI/ClawRouter
- **License**: MIT
- **Version**: 0.10.5
- **Runtime**: Node.js >=20, TypeScript
- **npm**: `@blockrun/clawrouter`

Primary goals:
- **Cost optimization**: Up to 92% savings by routing to cheapest capable model
- **Agent-native payment**: x402 micropayments (USDC on Base blockchain)
- **Zero-latency routing**: All decisions local, <1ms
- **Non-custodial**: User funds stay in their wallet until spent

### 2.2 Architecture

Local proxy server (default port 8402) with a request processing pipeline:

```
OpenClaw Client
    │
    ▼
localhost:8402/v1/chat/completions
    │
    ├── Request Deduplication (SHA-256, 30s TTL)
    ├── Smart Router (14-dimension scorer, <1ms)
    ├── Balance Check (cached USDC on Base)
    ├── x402 Payment Flow (EIP-712 signed)
    ├── Provider API Call (with fallback chain)
    ├── Response Cache (200 entries, 10min TTL, LRU)
    └── SSE Streaming (2s heartbeats)
```

Key source files:

| File | Purpose |
|------|---------|
| `proxy.ts` | HTTP proxy server, pipeline orchestration |
| `router/index.ts` | Router orchestration |
| `router/rules.ts` | 14-dimension weighted classifier |
| `router/selector.ts` | Tier-to-model mapping |
| `router/llm-classifier.ts` | LLM fallback classifier |
| `router/config.ts` | Default weights and keywords |
| `x402.ts` | x402 payment protocol |
| `balance.ts` | On-chain balance monitoring |
| `dedup.ts` | Request deduplication |
| `response-cache.ts` | LRU response cache |
| `models.ts` | 41+ model definitions with pricing |
| `session.ts` | Session pinning |
| `journal.ts` | Session memory/action journal |
| `retry.ts` | Exponential backoff with fallback |
| `compression/` | Auto-compression for large contexts |

### 2.3 Two-Stage Classification System

#### Stage 1: Rule-Based Classifier (~80% of requests, <1ms)

The `classifyByRules()` function evaluates requests across 14 weighted dimensions, producing a score that maps to complexity tiers.

**Confidence calibration** uses a sigmoid function on the distance from the nearest tier boundary. Below 0.7 confidence → ambiguous → eligible for LLM fallback.

**Special overrides**:
- 2+ reasoning keywords in user prompt → force REASONING tier (85%+ confidence)
- Context exceeding token threshold → force COMPLEX tier
- Structured output requests → minimum MEDIUM tier
- `tools[]` array present → agentic tier models

#### Stage 2: LLM Fallback Classifier (~20% of requests)

Designed for ambiguous requests where rule-based confidence is low. Uses a cheap model (Gemini 2.5 Flash, ~$0.00003 per call). Currently defaults to configurable tier (MEDIUM) without actual LLM call.

### 2.4 The 14 Scoring Dimensions

| # | Dimension | Weight | Signal |
|---|-----------|--------|--------|
| 1 | Token count | 0.08 | Short vs. long inputs |
| 2 | Code presence | 0.15 | Keywords: function, class, import, async |
| 3 | Reasoning markers | 0.18 | Keywords: prove, theorem, derive, chain of thought |
| 4 | Technical terms | 0.10 | algorithm, optimize, kubernetes |
| 5 | Creative markers | 0.05 | story, poem, brainstorm |
| 6 | Simple indicators | 0.02 | "what is", "define", "translate" (negative) |
| 7 | Multi-step patterns | 0.12 | Regex: "first...then", "step 1" |
| 8 | Question complexity | 0.05 | Count of question marks |
| 9 | Imperative verbs | 0.03 | Action word presence |
| 10 | Constraint indicators | 0.04 | Constraint language |
| 11 | Output format | 0.03 | JSON/YAML format requests |
| 12 | Reference complexity | 0.02 | Cross-references |
| 13 | Negation complexity | 0.01 | Negation patterns |
| 14 | Domain specificity | 0.02 | Specialized domain terms |

Plus **agentic task score** (weight 0.04): file operations, test execution, debugging loops.

### 2.5 Complexity Tiers

| Tier | Score Range | Example Tasks | Typical Model |
|------|-------------|---------------|---------------|
| SIMPLE | < 0.0 | Factual Q&A, definitions | Cheap/free model |
| MEDIUM | 0.0 – 0.3 | Summaries, moderate code | Mid-tier model |
| COMPLEX | 0.3 – 0.5 | Multi-step code, system design | Capable model |
| REASONING | > 0.5 | Proofs, formal logic, analysis | Frontier model |

### 2.6 Payment Innovation (x402)

ClawRouter's most distinctive feature is its payment model using the x402 protocol:

1. Client sends request to provider
2. Provider responds with HTTP 402 (Payment Required) + pricing info
3. Client signs an EIP-712 `TransferWithAuthorization` for USDC on Base
4. Client retries with payment header
5. Provider verifies payment and serves the response

**Pre-authorization optimization**: After the first 402 handshake, payment parameters are cached. Subsequent requests pre-sign and skip the round trip (~200ms savings).

**Graceful degradation**: Empty wallet → automatic fallback to free-tier models.

> **Note**: This payment model is **not adopted** in our combined project. It is documented here for research completeness.

### 2.7 Key Design Patterns

1. **Two-Layer Classification with Confidence Gating** — Fast rule-based path with LLM fallback for ambiguous cases
2. **Session Pinning** — Multi-turn conversations locked to initially selected model
3. **Session Journal / Memory** — Regex-based action extraction from responses for cross-turn context
4. **Request Deduplication** — SHA-256 with JSON canonicalization and timestamp stripping
5. **Optimistic Balance Deduction** — Local balance tracking to avoid RPC calls
6. **Fallback Chain with Context-Aware Filtering** — Models filtered by context window before fallback
7. **Pre-Authorization Caching** — Skip 402 round trip after first handshake
8. **Graceful Degradation** — Empty wallet → free tier instead of failure
9. **Auto-Compression** — Requests >180KB compressed before routing

### 2.8 Tech Stack

| Component | Technology |
|-----------|-----------|
| Runtime | Node.js >=20 |
| Language | TypeScript 5.7+ |
| Blockchain | viem 2.39+ (Ethereum/Base) |
| Bundler | tsup |
| Tests | Vitest |
| Framework | OpenClaw plugin system |

Supported providers (7): OpenAI, Anthropic, Google, DeepSeek, xAI, Moonshot, MiniMax, NVIDIA

### 2.9 Unique Innovations

1. **x402 Protocol**: HTTP 402 as a native LLM payment mechanism (blockchain micropayments replacing API keys)
2. **Agent-Native Commerce**: Autonomous agents generating wallets and paying for their own compute
3. **14-Dimension Sigmoid-Calibrated Scoring**: Multi-signal complexity classification with confidence estimation
4. **Agentic Auto-Detection**: Identifying multi-step autonomous workflows for model optimization
5. **AI-Powered Self-Diagnostics**: The `doctor` command uses its own payment system to analyze diagnostic output
6. **Multilingual Keyword Support**: Scoring keywords span EN, ZH, JA, RU, DE, ES, PT, KO, AR

---

## 3. Comparative Analysis

| Dimension | semantic-router | ClawRouter |
|-----------|----------------|------------|
| **Primary language** | Go (43%) + Rust (17%) + Python (20%) | TypeScript (100%) |
| **Deployment model** | Envoy ExtProc sidecar (cloud/datacenter) | Local Node.js proxy (dev machine) |
| **Scale target** | Cloud/datacenter/edge, multi-tenant | Single developer, single machine |
| **Classification** | Neural (BERT) + heuristic, parallel signals | Rule-based (14-dim weighted) + LLM fallback |
| **Classification latency** | ~10-50ms (neural inference) | <1ms (pure heuristic) |
| **Selection algorithms** | 12+ (Elo, RL, RouterDC, AutoMix, ML-based) | Tier-to-model lookup table |
| **Security** | Jailbreak, PII, hallucination detection | None (delegates to providers) |
| **Caching** | Semantic (similarity-based, HNSW) | Response cache (LRU, exact match) |
| **Payment** | Standard API keys / provider auth | x402 blockchain micropayments |
| **Multi-model** | Confidence escalation, ratings, ReMoM, RL | Single model per tier + fallback chain |
| **Kubernetes** | Full CRDs, operator, Helm charts | None |
| **Memory** | Milvus-backed agentic memory | Regex-based action journal |
| **Observability** | Prometheus + OpenTelemetry + structured logging | ASCII stats display |
| **Session management** | None (stateless per-request) | Model pinning per conversation |
| **Deduplication** | None | SHA-256 content hashing |
| **Compression** | None | Auto-compress >180KB |
| **Maturity** | 3.2k stars, 73 contributors, NeurIPS paper | Early stage, single team |

### Key Takeaway

semantic-router is **infrastructure**: deep, production-hardened, Kubernetes-native, with sophisticated ML-based classification and selection. ClawRouter is **developer experience**: fast, local, cost-aware, with clever lightweight heuristics. The combined project takes semantic-router's architecture and enriches it with ClawRouter's practical optimizations.

---

## 4. Synthesis: What We Take from Each

### From semantic-router (Primary Architecture)
- Envoy ExtProc integration model
- Go/Rust core implementation
- Filter chain pipeline pattern
- Signal-based decision engine with boolean expression trees
- 12+ model selection algorithms
- Security pipeline (jailbreak, PII, hallucination)
- Semantic caching with HNSW
- Multi-model execution strategies (looper)
- Agentic memory system
- MCP integration
- Kubernetes CRDs and operator
- Full observability stack
- Training pipelines
- CLI tooling

### From ClawRouter (Influenced By)
- **Fast-path pre-classifier**: 14-dimension weighted scorer for <1ms classification of obvious requests
- **Session pinning**: Multi-turn conversation model consistency
- **Request deduplication**: SHA-256 content hashing to prevent duplicate inference
- **Context auto-compression**: Automatic compression for large contexts
- **Agentic task detection**: Signal type for identifying autonomous multi-step workflows
- **Graceful degradation**: Budget-exhaustion fallback to cheaper models instead of failure
- **Context-aware fallback chains**: Filtering fallback models by context window capacity
- **Confidence gating**: Sigmoid calibration for routing to fast vs. full classification paths

### Explicitly NOT Adopted
- x402 blockchain payment protocol (too niche for core infrastructure)
- TypeScript implementation (Go/Rust is the correct choice for infrastructure)
- OpenClaw plugin interface (too tightly coupled to one client)
- Hardcoded model catalog (dynamic config is superior)
- Local-only proxy model (Envoy ExtProc is the integration point)
