# Skills: Semantic Router Capabilities Reference

> Complete reference of the router's capabilities, organized by category.
> Each skill describes what the router can do, how it works, and how to configure it.

---

## Table of Contents

- [Routing Skills](#routing-skills)
- [Classification Skills](#classification-skills)
- [Selection Skills](#selection-skills)
- [Security Skills](#security-skills)
- [Performance Skills](#performance-skills)
- [Resilience Skills](#resilience-skills)
- [Observability Skills](#observability-skills)
- [Integration Skills](#integration-skills)
- [Memory & Context Skills](#memory--context-skills)
- [Deployment Skills](#deployment-skills)

---

## Routing Skills

### Semantic Request Routing
**What**: Routes LLM requests to the optimal backend model based on content analysis, not just load balancing.

**How**: Extracts multiple signals from each request (domain, complexity, intent, language, modality), evaluates them against configurable decision rules using boolean expression trees (AND/OR/NOT), and selects the best-matching routing decision.

**Config**:
```yaml
decisions:
  - name: "code-generation"
    rules:
      type: "AND"
      conditions:
        - signal: "domain"
          category: "coding"
        - signal: "complexity"
          level: "high"
    model: "deepseek-coder-v3"
```

---

### Multi-Provider Routing
**What**: Routes requests across multiple LLM providers with automatic format translation.

**How**: Maintains provider profiles for each backend. When routing to non-OpenAI providers (Anthropic, Gemini, etc.), automatically translates request/response formats and injects provider-specific authentication headers.

**Supported Providers**:
| Provider | Format | Auth |
|----------|--------|------|
| vLLM | OpenAI-compatible | Header injection |
| OpenAI | Native | API key |
| Anthropic | Translated from OpenAI | API key |
| Azure OpenAI | OpenAI + Azure headers | Azure AD / API key |
| AWS Bedrock | Translated | IAM / SigV4 |
| Google Gemini | Translated | API key |
| Google Vertex AI | Translated | Service account |
| DeepSeek | OpenAI-compatible | API key |

**Config**:
```yaml
providerProfiles:
  - name: "anthropic-prod"
    baseURL: "https://api.anthropic.com"
    authHeader: "x-api-key"
    chatPath: "/v1/messages"
```

---

### LoRA Adapter Routing
**What**: Routes requests to specific LoRA adapters served by vLLM instances.

**How**: Each Decision can specify a LoRA adapter name. The router modifies the model name in the request to target the appropriate adapter on the vLLM server.

**Config**:
```yaml
decisions:
  - name: "legal-docs"
    model: "llama-3-70b"
    loraAdapter: "legal-lora-v2"
```

---

### Image Generation Routing
**What**: Routes image generation requests to appropriate diffusion model backends.

**How**: Modality detection identifies requests for image generation (AR vs. Diffusion) and routes them to dedicated backends optimized for visual content.

---

## Classification Skills

### Fast-Path Pre-Classification
**What**: Classifies 70-80% of requests in <1ms using lightweight heuristics, bypassing expensive neural inference.

**Influenced by**: ClawRouter's 14-dimension weighted scoring system.

**How**: Evaluates request content across 14 weighted dimensions (code presence, reasoning markers, technical terms, multi-step patterns, etc.), computes a weighted sum, maps to complexity tiers, and calibrates confidence using a sigmoid function. High-confidence classifications skip the full neural pipeline.

**Dimensions** (15 total):

| Dimension | Weight | What It Detects |
|-----------|--------|----------------|
| Reasoning markers | 0.18 | `prove`, `theorem`, `derive`, `chain of thought` |
| Code presence | 0.15 | `function`, `class`, `import`, `async`, `def` |
| Multi-step patterns | 0.12 | `first...then`, `step 1`, sequential instructions |
| Technical terms | 0.10 | `algorithm`, `optimize`, `kubernetes`, `distributed` |
| Token count | 0.08 | Short (<100 tokens) vs. long (>2000 tokens) inputs |
| Creative markers | 0.05 | `story`, `poem`, `brainstorm`, `imagine` |
| Question complexity | 0.05 | Multiple question marks, nested questions |
| Constraint indicators | 0.04 | `must`, `ensure`, `constraint`, `within N` |
| Agentic task | 0.04 | `tools[]` present, file ops, test-debug loops |
| Imperative verbs | 0.03 | `implement`, `design`, `analyze`, `build` |
| Output format | 0.03 | `json`, `yaml`, `csv`, `table`, `markdown` |
| Simple indicators | 0.02 | `what is`, `define`, `translate` (negative score) |
| Reference complexity | 0.02 | Cross-references, `as mentioned above` |
| Domain specificity | 0.02 | Specialized vocabulary (medical, legal, etc.) |
| Negation complexity | 0.01 | `not`, `without`, `except`, complex negations |

**Tiers**:
| Tier | Score Range | Typical Routing |
|------|-------------|-----------------|
| SIMPLE | < 0.0 | Cheapest capable model |
| MEDIUM | 0.0 – 0.3 | Mid-tier model |
| COMPLEX | 0.3 – 0.5 | Capable model |
| REASONING | > 0.5 | Frontier model |

**Config**:
```yaml
fastPathClassifier:
  enabled: true
  confidenceThreshold: 0.7
  weights:
    reasoningMarkers: 0.18
    codePresence: 0.15
    multiStepPatterns: 0.12
    # ... all 15 dimensions configurable
  tierBoundaries:
    simple: 0.0
    medium: 0.3
    complex: 0.5
  tierToDecisionMapping:
    SIMPLE: "simple-queries"
    MEDIUM: "general-purpose"
    COMPLEX: "complex-tasks"
    REASONING: "deep-reasoning"
```

---

### Neural Domain Classification
**What**: Classifies requests into domain categories using fine-tuned BERT/ModernBERT models.

**How**: Runs BERT-family models natively via Rust/Candle bindings (no Python runtime). Supports standard BERT, ModernBERT, and mmBERT-32K for long-context classification.

**Config**:
```yaml
categories:
  - name: "coding"
    description: "Software development tasks"
  - name: "legal"
    description: "Legal analysis and document review"
  - name: "medical"
    description: "Healthcare and medical queries"
```

---

### Embedding Similarity
**What**: Routes requests based on vector similarity to pre-defined clusters or examples.

**How**: Generates embeddings using Qwen3, Gemma, mmBERT, or standard BERT models. Compares against pre-computed reference embeddings using HNSW index with SIMD-optimized distance computation.

---

### Keyword Matching
**What**: Pattern-based routing using regex, BM25, and n-gram matching.

**How**: Three matching strategies:
- **Regex**: Full regular expression matching on message content
- **BM25**: Term frequency-inverse document frequency relevance scoring
- **N-gram**: Character and word n-gram matching for fuzzy matching

---

### Language Detection
**What**: Identifies the language of incoming requests.

**How**: Combines heuristic character-set analysis with neural classification for ambiguous cases.

---

### Complexity Assessment
**What**: Multi-factor estimation of task complexity.

**How**: Combines token count, vocabulary diversity, structural complexity (nested instructions, constraints), and reference patterns to estimate task difficulty.

---

### Modality Detection
**What**: Identifies whether a request targets text generation (autoregressive) or image generation (diffusion).

**How**: Analyzes request format, model name patterns, and content indicators.

---

### Agentic Task Detection
**What**: Identifies multi-step autonomous workflows that require agent-optimized models.

**Influenced by**: ClawRouter's agentic auto-detection concept.

**How**: Detects tool-use patterns (`tools[]` array), file operation keywords, shell commands, iterative debugging patterns, and multi-step planning language. Outputs a score indicating agentic task likelihood.

**Detected patterns**:
- Tool use: `tools[]` array in request
- File operations: `read_file`, `write_file`, `create`, `edit`, `delete`
- Shell operations: `run_command`, `execute`, `bash`, `terminal`
- Iterative work: `try again`, `retry`, `fix`, `debug`, `test and fix`
- Multi-step planning: `step 1`, `first...then`, `plan`, `implement`

**Config**:
```yaml
signals:
  - type: "agentic"
    minScore: 0.6
    taskTypes: ["coding", "research", "automation"]
```

---

## Selection Skills

### Static Model Selection
**What**: Deterministic model assignment based on configuration scoring.

**Best for**: Predictable deployments with well-known workload patterns.

---

### Elo Rating Selection
**What**: Bradley-Terry rating system with time decay for adaptive model selection.

**How**: Maintains Elo ratings for each model based on user feedback and quality signals. Ratings decay over time to adapt to model updates.

**Best for**: Quality-focused selection that improves over time.

---

### RouterDC Selection
**What**: Dual-contrastive learning embeddings for model-request matching.

**How**: Learns embeddings that place models close to requests they excel at, using contrastive pairs from historical data.

**Best for**: High-volume systems with diverse request types.

---

### AutoMix Selection
**What**: POMDP-based cost-quality optimization for budget-constrained selection.

**How**: Models the selection problem as a Partially Observable Markov Decision Process, optimizing for the Pareto frontier of cost vs. quality.

**Best for**: Budget-constrained deployments maximizing quality per dollar.

---

### RL-Driven Selection
**What**: Thompson Sampling with personalization for exploration/exploitation balanced selection.

**How**: Maintains Bayesian estimates of each model's quality for different request types. Uses Thompson Sampling to balance exploring new model-request pairings with exploiting known good matches.

**Best for**: Dynamic environments with changing model capabilities.

---

### Hybrid Selection
**What**: Weighted combination of multiple selection algorithms.

**How**: Runs multiple selectors and combines their scores with configurable weights.

**Best for**: Balanced workloads requiring multiple optimization objectives.

---

### ML-Based Selection (KNN, KMeans, SVM, MLP)
**What**: Trained classifiers for model selection based on historical routing data.

**How**: Models trained offline on historical request-model-outcome data. Loaded at startup for inference-time selection.

**Best for**: Deployments with large historical datasets.

---

### GMTRouter Selection
**What**: Graph-based personalized routing using user-preference graphs.

**Best for**: Multi-tenant deployments with per-user optimization.

---

### Latency-Aware Selection
**What**: Model selection optimized for TPOT and TTFT metrics.

**How**: Monitors per-model latency percentiles and routes latency-sensitive requests to the fastest available models.

**Best for**: Latency-critical applications (chatbots, real-time assistants).

---

## Security Skills

### Jailbreak Detection
**What**: Detects and blocks adversarial inputs designed to bypass model safety mechanisms.

**How**: Runs PromptGuard (BERT-based) via native Rust/Candle inference. Configurable confidence threshold per routing decision.

**Latency**: ~5ms (Candle) or ~20ms (vLLM backend)

**Config**:
```yaml
decisions:
  - name: "customer-facing"
    plugins:
      jailbreak:
        enabled: true
        threshold: 0.85
        action: "block"  # or "log"
```

---

### PII Detection & Redaction
**What**: Identifies and handles personally identifiable information in requests.

**How**: Token-level NER classification via Candle. Detects names, emails, phone numbers, SSNs, credit card numbers, and addresses.

**Actions**: `block` (reject request), `redact` (replace with `[REDACTED]`), `log` (continue with warning)

**Config**:
```yaml
decisions:
  - name: "internal-tools"
    plugins:
      pii:
        enabled: true
        action: "redact"
        entityTypes: ["email", "phone", "ssn", "credit_card"]
```

---

### Hallucination Detection
**What**: Detects factual inconsistencies in model responses.

**How**: Response-phase NLI (Natural Language Inference) checking against source documents provided by RAG or conversation history.

**Config**:
```yaml
decisions:
  - name: "research-assistant"
    plugins:
      hallucination:
        enabled: true
        action: "flag"  # or "block", "warn"
```

---

### Rate Limiting
**What**: Enforces request quotas per user, per decision, or globally.

**How**: Integrates with Envoy Rate Limit Service (RLS) for distributed limiting, with local limiter fallback. Enhanced with degradation policy (see Resilience Skills).

---

## Performance Skills

### Semantic Caching
**What**: Caches responses based on semantic similarity, not exact match.

**How**: Generates embeddings for requests and stores in HNSW index. Similar future requests (above configurable threshold) receive cached responses without backend calls.

**Backends**: In-memory (development), Redis (multi-instance), Milvus (large-scale)

**Config**:
```yaml
decisions:
  - name: "faq-queries"
    plugins:
      cache:
        enabled: true
        similarityThreshold: 0.95
        ttl: 3600
        backend: "redis"
```

---

### Request Deduplication
**What**: Prevents duplicate inference when clients retry after timeouts.

**Influenced by**: ClawRouter's SHA-256 deduplication system.

**How**: Canonicalizes request JSON (sort keys, strip timestamps/request IDs), computes SHA-256 hash, checks time-windowed LRU cache. Duplicate requests receive the cached response; in-flight duplicates wait for the original to complete.

**Config**:
```yaml
deduplication:
  enabled: true
  windowSeconds: 30
  maxEntries: 10000
  hashAlgorithm: "sha256"
  stripFields:
    - "timestamp"
    - "request_id"
```

---

### Context Auto-Compression
**What**: Automatically compresses large request contexts to reduce token usage and cost.

**Influenced by**: ClawRouter's auto-compression for contexts exceeding 180KB.

**How**: Three compression strategies applied sequentially:
1. **Whitespace normalization**: Collapse whitespace runs, normalize line endings (5-15% savings)
2. **Content deduplication**: Remove repeated text blocks (10-40% savings, especially after RAG injection)
3. **JSON compaction**: Minify JSON/YAML in code blocks and tool responses (10-30% savings)

**Config**:
```yaml
contextCompression:
  enabled: true
  thresholdBytes: 184320  # 180KB
  strategies:
    - "whitespace"
    - "dedup"
    - "json-compact"
```

---

### SIMD-Optimized Vector Search
**What**: Hardware-accelerated similarity search for semantic caching and classification.

**How**: Custom HNSW implementation with AMD64 assembly for cosine distance and dot product computation using SIMD instructions.

---

## Resilience Skills

### Session Pinning
**What**: Keeps multi-turn conversations on the same model for consistency.

**Influenced by**: ClawRouter's session pinning concept.

**How**: Generates a conversation fingerprint from the first message (or conversation ID header). Pins the selected model and endpoint for configurable duration. Re-pins if the pinned model becomes unavailable.

**Config**:
```yaml
decisions:
  - name: "conversational"
    plugins:
      sessionPinning:
        enabled: true
        ttlMinutes: 60
        storage: "redis"  # or "memory"
        fingerprintSources:
          - "first_message"
          - "api_key"
```

---

### Graceful Degradation
**What**: Automatically falls back to cheaper models instead of failing when resources are exhausted.

**Influenced by**: ClawRouter's "empty wallet → free tier" concept.

**How**: When rate limits are hit or budgets are exceeded, the router re-routes to a configured fallback model pool instead of returning errors. Custom headers inform clients that a degraded model was used.

**Config**:
```yaml
decisions:
  - name: "production-api"
    plugins:
      degradationPolicy:
        enabled: true
        triggers: ["rate_limit", "budget_exceeded", "provider_error"]
        fallbackModel: "nvidia/gpt-oss-120b"
        notifyHeaders:
          X-Degraded: "true"
          X-Original-Model: "{{requested_model}}"
```

---

### Context-Aware Fallback Chains
**What**: Ensures fallback models can actually handle the request's context size.

**Influenced by**: ClawRouter's context-length-aware fallback filtering.

**How**: Each model in the fallback chain is filtered by whether its context window can accommodate the request size (with 10% buffer). Models with insufficient context windows are skipped.

**Config**:
```yaml
backendModels:
  - name: "gpt-4o"
    contextWindow: 128000
    fallback: ["claude-sonnet-4", "gemini-2.5-pro", "deepseek-chat"]
```

---

### Multi-Model Execution
**What**: Sends requests to multiple models for quality improvement.

**Strategies**:
| Strategy | Description |
|----------|-------------|
| Confidence | Escalate through models until quality threshold is met |
| Ratings | Parallel execution, select best by quality rating |
| ReMoM | Multi-round refinement across models |
| RL-Driven | RL agent selects optimal model sequence |

---

### Hot Configuration Reload
**What**: Updates routing configuration without downtime or dropped requests.

**How**: Uses `atomic.Pointer` for lock-free reads during request processing. Configuration file watcher detects changes, builds new router instance, and atomically swaps the pointer.

---

## Observability Skills

### Prometheus Metrics
**What**: Comprehensive time-series metrics for monitoring and alerting.

**Key metrics**:
- Request count by decision, model, status
- Latency histograms (total, TTFT, TPOT)
- Fast-path classification rates and bypass counts
- Dedup hit rates
- Session pinning events
- Compression savings
- Degradation activations
- Cache hit rates
- Security blocks
- Cost tracking by model and decision

---

### Distributed Tracing
**What**: End-to-end request tracing across the routing pipeline.

**How**: OpenTelemetry integration with Jaeger and Zipkin exporters. Creates spans for each pipeline filter stage, classifier inference, cache lookups, and backend calls.

---

### Structured Logging
**What**: JSON-formatted logs with full request context.

**Fields**: Request ID, trace ID, decision name, selected model, classification signals, security findings, latency breakdown.

---

### Cost Tracking
**What**: Real-time tracking of inference cost by model, decision, and user.

**How**: Computes cost from token usage and model pricing. Reports via Prometheus metrics and optional cost headers in responses.

---

## Integration Skills

### Envoy ExtProc
**What**: Native integration with Envoy proxy as an external processor.

**How**: Implements the Envoy `ext_proc` gRPC service. Envoy consults the router for every request/response passing through the proxy, enabling transparent insertion without client or backend changes.

---

### Model Context Protocol (MCP)
**What**: Standardized tool integration via the MCP specification.

**How**: MCP client supporting both stdio (local tool servers) and HTTP (remote tool servers) transports. Enables tool discovery, invocation, resource reading, and prompt retrieval.

---

### Kubernetes CRDs
**What**: Kubernetes-native configuration using Custom Resource Definitions.

**Resources**: `IntelligentPool`, `IntelligentRoute`, `SemanticRouter`

---

### RAG Backends
**What**: Multiple Retrieval-Augmented Generation backends for context enrichment.

**Backends**:
| Backend | Use Case |
|---------|----------|
| Cache-based | Re-use cached context from similar queries |
| External service | Dedicated RAG API endpoint |
| Hybrid | Combine multiple sources |
| MCP-based | RAG via Model Context Protocol tools |
| Milvus | Direct vector store queries |
| OpenAI | OpenAI file/vector store APIs |
| Generic vector store | Any vector database |

---

### OpenAI API Compatibility
**What**: Full compatibility with the OpenAI chat completion API format.

**Endpoints**: `/v1/chat/completions`, `/v1/models`, `/v1/responses`

---

## Memory & Context Skills

### Agentic Cross-Session Memory
**What**: Persistent memory across conversations for agentic workflows.

**How**: Extracts key facts, actions, and decisions from responses. Stores in Milvus vector database. Retrieves relevant memories for new requests based on semantic similarity.

**Features**: Deduplication, time-decay scoring, per-user/per-session scoping.

---

### Response API (Stateful Conversations)
**What**: Stateful conversation management for multi-turn interactions.

**How**: Stores conversation history and state. Supports `/v1/responses` endpoint format.

**Backends**: In-memory, Milvus, Redis.

---

### Router Replay
**What**: Records routing decisions for debugging, auditing, and training.

**How**: Stores the full decision context (signals, scores, selected model, response quality) for later analysis.

**Backends**: In-memory, Redis, PostgreSQL, Milvus.

---

## Deployment Skills

### Helm Chart
**What**: Production-ready Kubernetes deployment via Helm.

**Includes**: Namespace, ServiceAccount, ConfigMap, PVC, Deployment, Service, Ingress, HPA. Separate values files for dev and production.

---

### Kubernetes Operator
**What**: Automated lifecycle management via a Kubebuilder-based operator.

**Manages**: SemanticRouter custom resources with reconciliation for configuration changes, scaling, and health monitoring.

---

### CLI Tool (vllm-sr)
**What**: Developer CLI for local development and management.

**Commands**: `init`, `serve`, `config`, `dashboard`, `status`, `logs`, `stop`

---

### Docker Support
**What**: Containerized deployment with multi-stage builds.

**Features**: Supervisord for process management, health checks, graceful shutdown.

---

### Dashboard
**What**: Web-based monitoring UI.

**Features**: Real-time metrics, request logs, decision visualization, model performance comparison.
