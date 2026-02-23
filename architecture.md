# Architecture: Semantic Router

> Detailed architectural specification for the combined intelligent LLM router.
> Primary lineage: vllm-project/semantic-router | Influenced by: BlockRunAI/ClawRouter

---

## Table of Contents

- [1. System Context](#1-system-context)
- [2. High-Level Architecture](#2-high-level-architecture)
- [3. Component Inventory](#3-component-inventory)
- [4. Request Pipeline (Detailed)](#4-request-pipeline-detailed)
- [5. Response Pipeline (Detailed)](#5-response-pipeline-detailed)
- [6. Fast-Path Pre-Classifier](#6-fast-path-pre-classifier)
- [7. Signal Classification System](#7-signal-classification-system)
- [8. Decision Engine](#8-decision-engine)
- [9. Model Selection](#9-model-selection)
- [10. Session Pinning](#10-session-pinning)
- [11. Request Deduplication](#11-request-deduplication)
- [12. Context Compression](#12-context-compression)
- [13. Agentic Task Detection](#13-agentic-task-detection)
- [14. Graceful Degradation](#14-graceful-degradation)
- [15. Security Pipeline](#15-security-pipeline)
- [16. Caching Architecture](#16-caching-architecture)
- [17. Multi-Model Execution](#17-multi-model-execution)
- [18. Memory System](#18-memory-system)
- [19. Configuration Architecture](#19-configuration-architecture)
- [20. Kubernetes Integration](#20-kubernetes-integration)
- [21. Observability](#21-observability)
- [22. Data Flow Diagrams](#22-data-flow-diagrams)
- [23. Module Dependency Map](#23-module-dependency-map)
- [24. Performance Targets](#24-performance-targets)

---

## 1. System Context

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          External Systems                               │
│                                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │ Client   │  │ OpenAI   │  │Anthropic │  │  vLLM    │  │ Gemini/  │ │
│  │   Apps   │  │   API    │  │   API    │  │ Clusters │  │ Others   │ │
│  └────┬─────┘  └────▲─────┘  └────▲─────┘  └────▲─────┘  └────▲─────┘ │
│       │              │              │              │              │      │
│       ▼              └──────┬───────┴──────┬───────┴──────┬───────┘      │
│  ┌─────────┐          ┌────┴───────────────┴──────────────┴────┐        │
│  │  Envoy  │◄────────►│         SEMANTIC ROUTER CORE           │        │
│  │  Proxy  │ ExtProc  │      (Go + Rust/Candle bindings)       │        │
│  └─────────┘          └────────────────┬───────────────────────┘        │
│                                        │                                 │
│                         ┌──────────────┼──────────────┐                  │
│                         ▼              ▼              ▼                  │
│                    ┌─────────┐   ┌──────────┐   ┌──────────┐            │
│                    │  Milvus │   │  Redis   │   │Prometheus│            │
│                    │(vectors)│   │ (cache)  │   │(metrics) │            │
│                    └─────────┘   └──────────┘   └──────────┘            │
└─────────────────────────────────────────────────────────────────────────┘
```

The router operates as an **Envoy External Processor (ExtProc)** — a gRPC service that Envoy consults for every request/response passing through the proxy. This architecture means:
- **No client changes**: Clients talk to Envoy using standard OpenAI-compatible APIs
- **No backend changes**: Backends receive standard requests
- **Wire-speed processing**: The router sees raw request/response bytes at the proxy layer
- **Deployment flexibility**: Can be deployed as a sidecar, daemonset, or standalone service

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SEMANTIC ROUTER CORE (Go)                         │
│                                                                     │
│  ┌────────────────────── Request Pipeline ───────────────────────┐  │
│  │                                                                │  │
│  │  ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌────────────┐  │  │
│  │  │  Dedup  │──►│ FastPath │──►│  Signal  │──►│  Decision  │  │  │
│  │  │ Filter  │   │Classifier│   │Extraction│   │  Engine    │  │  │
│  │  └─────────┘   └──────────┘   └──────────┘   └────────────┘  │  │
│  │       │              │              │               │          │  │
│  │       │         [skip if high       │               │          │  │
│  │       │          confidence]        │               ▼          │  │
│  │       │              │         ┌────┴────┐   ┌────────────┐   │  │
│  │       │              └────────►│Jailbreak│──►│    PII     │   │  │
│  │       │                        │  Guard  │   │  Filter    │   │  │
│  │       │                        └─────────┘   └────────────┘   │  │
│  │       │                                           │           │  │
│  │       │    ┌──────────┐   ┌──────────┐   ┌───────┴──────┐    │  │
│  │       │    │  Memory  │◄──│   RAG    │◄──│ Cache Lookup │    │  │
│  │       │    │Retrieval │   │Injection │   │              │    │  │
│  │       │    └──────────┘   └──────────┘   └──────────────┘    │  │
│  │       │         │                                             │  │
│  │       │    ┌────┴─────┐   ┌──────────┐   ┌──────────────┐    │  │
│  │       │    │ Context  │──►│ Session  │──►│    Model     │    │  │
│  │       │    │Compress  │   │  Pinning │   │  Selection   │    │  │
│  │       │    └──────────┘   └──────────┘   └──────────────┘    │  │
│  │       │                        │               │              │  │
│  │       │                        │          ┌────┴──────┐       │  │
│  │       │                        │          │ Provider  │       │  │
│  │       │                        │          │ Routing   │       │  │
│  │       │                        │          └───────────┘       │  │
│  └───────┼────────────────────────┼──────────────────────────────┘  │
│          │                        │                                  │
│  ┌───────┼──────────── Response Pipeline ────────────────────────┐  │
│  │       │                        │                               │  │
│  │  ┌────┴─────┐  ┌──────────┐  ┌┴─────────┐  ┌──────────────┐  │  │
│  │  │  Dedup   │  │ Latency  │  │ Session  │  │Hallucination │  │  │
│  │  │  Update  │  │ Metrics  │  │ Update   │  │  Detection   │  │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────┘  │  │
│  │       │              │             │              │            │  │
│  │  ┌────┴─────┐  ┌────┴─────┐  ┌────┴─────┐  ┌────┴─────┐     │  │
│  │  │   Cost   │  │  Cache   │  │  Memory  │  │  Rate    │     │  │
│  │  │ Tracking │  │ Populate │  │ Extract  │  │ Report   │     │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─── ML Inference ───┐  ┌── Config ──┐  ┌── Observability ─────┐  │
│  │  Rust/Candle:       │  │ YAML       │  │ Prometheus metrics   │  │
│  │  - BERT classifiers │  │ K8s CRDs   │  │ OpenTelemetry traces │  │
│  │  - PromptGuard      │  │ Env vars   │  │ Structured logging   │  │
│  │  - PII detector     │  │ Hot-reload │  │                      │  │
│  │  - Embeddings       │  │            │  │                      │  │
│  └─────────────────────┘  └────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Inventory

### Core Packages (`src/semantic-router/pkg/`)

| Package | Lineage | Description |
|---------|---------|-------------|
| `extproc/` | semantic-router | Envoy ExtProc gRPC service, request/response filter chains |
| `decision/` | semantic-router | Boolean expression tree evaluator over signal matches |
| `classification/` | semantic-router + ClawRouter | Multi-signal classifiers (neural + heuristic + agentic) |
| `selection/` | semantic-router + ClawRouter | Model selection algorithms (12+) with degradation policy |
| `fastpath/` | **ClawRouter-influenced** | 14-dimension weighted pre-classifier (<1ms) |
| `session/` | **ClawRouter-influenced** | Conversation-to-model pinning with TTL |
| `dedup/` | **ClawRouter-influenced** | SHA-256 request deduplication with LRU cache |
| `compression/` | **ClawRouter-influenced** | Context auto-compression for large requests |
| `cache/` | semantic-router | Semantic caching (memory/Redis/Milvus) with HNSW |
| `hnsw/` | semantic-router | HNSW vector index with SIMD assembly optimizations |
| `looper/` | semantic-router | Multi-model execution strategies |
| `memory/` | semantic-router | Agentic cross-session memory (Milvus-backed) |
| `mcp/` | semantic-router | Model Context Protocol client (stdio + HTTP) |
| `k8s/` | semantic-router | Kubernetes CRD controller |
| `config/` | semantic-router | Configuration model and hot-reload |
| `openai/` | semantic-router | OpenAI API compatibility layer |
| `anthropic/` | semantic-router | Anthropic API translation |
| `authz/` | semantic-router | Credential resolution chains |
| `ratelimit/` | semantic-router | Rate limiting (Envoy RLS + local) |
| `observability/` | semantic-router | Logging, metrics, tracing |
| `responseapi/` | semantic-router | Stateful conversation support |
| `routerreplay/` | semantic-router | Decision recording and retrieval |
| `imagegen/` | semantic-router | Diffusion model routing |
| `latency/` | semantic-router | TPOT/TTFT metrics tracking |
| `vectorstore/` | semantic-router | Document ingestion and search |
| `tools/` | semantic-router | Tool routing (top-k retrieval) |
| `modelselection/` | semantic-router | Trained model loading (KNN, SVM, etc.) |

### Supporting Components

| Component | Location | Language | Purpose |
|-----------|----------|----------|---------|
| Candle Bindings | `candle-binding/` | Rust + Go (CGo) | Native BERT inference without Python |
| Training Pipelines | `src/training/` | Python | ML model training (12 pipelines) |
| CLI Tool | `src/vllm-sr/` | Python | Developer CLI (`vllm-sr`) |
| Dashboard | `src/dashboard/` | TypeScript | Web UI for monitoring |
| K8s Operator | `deploy/operator/` | Go | Kubebuilder-based CRD operator |
| Helm Chart | `deploy/helm/` | YAML | Kubernetes deployment |
| Envoy Config | `deploy/envoy/` | YAML | Proxy configuration |

---

## 4. Request Pipeline (Detailed)

The request body handler executes a sequential pipeline of filter stages. Each filter can:
- **Continue**: Pass to the next filter
- **Short-circuit**: Return a response directly (cache hit, security block, dedup hit)
- **Mutate**: Modify the request context (add headers, transform body, inject context)

### Stage 1: Request Deduplication
**Source**: `pkg/extproc/req_filter_dedup.go` (ClawRouter-influenced)

```
Input: Raw request body bytes
Process:
  1. Canonicalize JSON (sort keys, strip timestamps/request IDs)
  2. Compute SHA-256 hash
  3. Check LRU cache (configurable window, default 30s)
  4. If hit: return cached response immediately (short-circuit)
  5. If miss: store hash → pending, continue pipeline
Output: Continue or cached response
```

**Why this is first**: Dedup must run before any expensive processing. A duplicate request should be caught immediately, not after running neural classifiers.

### Stage 2: Request Parsing
**Source**: `pkg/extproc/req_filter_parse.go`

```
Input: Raw request body bytes
Process:
  1. Parse as OpenAI ChatCompletion request
  2. Extract: model name, messages, tools, stream flag, parameters
  3. Detect Response API format → translate if needed
  4. Populate RequestContext with parsed fields
Output: Populated RequestContext
```

### Stage 3: Fast-Path Pre-Classifier
**Source**: `pkg/extproc/req_filter_fastpath.go` (ClawRouter-influenced)

```
Input: RequestContext (parsed messages, model, tools)
Process:
  1. Run 14-dimension weighted scorer on request content
  2. Compute weighted sum across all dimensions
  3. Map to complexity tier (SIMPLE/MEDIUM/COMPLEX/REASONING)
  4. Calibrate confidence via sigmoid on distance from tier boundary
  5. If confidence >= threshold (default 0.7):
     a. Map tier to pre-configured Decision name
     b. Set decision in RequestContext
     c. Skip full signal extraction (jump to Stage 6)
  6. If confidence < threshold:
     a. Mark as "ambiguous" for full classification
     b. Continue to Stage 4
Output: Decision (high confidence) or continue to full classification
```

**Performance target**: <1ms p99 latency. This should handle 70-80% of requests.

### Stage 4: Full Signal Extraction (Parallel)
**Source**: `pkg/classification/unified_classifier.go`

Only runs if fast-path returned low confidence. Executes all signal extractors concurrently via goroutines:

```
Input: RequestContext
Process (parallel):
  ├── Keyword matching (regex, BM25, n-gram)
  ├── Embedding similarity (Qwen3/Gemma/BERT via Candle)
  ├── Domain classification (BERT/ModernBERT via Candle)
  ├── Complexity assessment (multi-factor)
  ├── Agentic detection (tools, patterns)  ◄── ClawRouter-influenced
  ├── Language detection
  ├── Modality detection (AR vs. Diffusion)
  ├── Fact-check necessity
  ├── User feedback classification
  └── Authorization rules
Output: SignalMatches struct with all signal results
```

### Stage 5: Decision Evaluation
**Source**: `pkg/decision/engine.go`

```
Input: SignalMatches
Process:
  1. For each configured Decision:
     a. Evaluate its rule tree (AND/OR/NOT over signal conditions)
     b. Compute confidence score
  2. selectBestDecision() picks winner by confidence then priority
Output: Selected Decision (or default)
```

### Stage 6: Jailbreak Detection
**Source**: `pkg/extproc/req_filter_jailbreak.go`

```
Input: RequestContext + selected Decision
Process:
  1. Check if Decision has jailbreak plugin enabled
  2. If enabled: run PromptGuard via Candle (or vLLM backend)
  3. If jailbreak detected above threshold: return 400 (short-circuit)
  4. If clean: continue
Output: Continue or block
```

### Stage 7: PII Detection
**Source**: `pkg/extproc/req_filter_pii.go`

```
Input: RequestContext + selected Decision
Process:
  1. Check if Decision has PII plugin enabled
  2. If enabled: run PII token classifier via Candle
  3. Based on Decision's PII policy:
     - "block": return 400 if PII found
     - "redact": replace PII tokens with [REDACTED]
     - "log": log PII detection, continue unchanged
Output: Continue (possibly with redacted body) or block
```

### Stage 8: Rate Limiting
**Source**: `pkg/extproc/req_filter_ratelimit.go`

```
Input: RequestContext + selected Decision
Process:
  1. Check rate limit quota (Envoy RLS or local limiter)
  2. If within quota: continue
  3. If exceeded:
     a. Check degradation policy (ClawRouter-influenced)
     b. If degradation enabled: re-route to fallback model pool
     c. If degradation disabled: return 429
Output: Continue (possibly with different model target) or 429
```

### Stage 9: Semantic Cache Lookup
**Source**: `pkg/extproc/req_filter_cache.go`

```
Input: RequestContext + selected Decision
Process:
  1. Check if Decision has cache plugin enabled
  2. Generate embedding for request content
  3. Query HNSW index for similar cached responses
  4. If similarity >= threshold: return cached response (short-circuit)
  5. If miss: continue
Output: Cached response or continue
```

### Stage 10: RAG Injection
**Source**: `pkg/extproc/req_filter_rag_*.go` (7 backends)

```
Input: RequestContext + selected Decision
Process:
  1. Query configured RAG backend(s) for relevant context
  2. Inject retrieved chunks into system/user messages
Output: RequestContext with augmented messages
```

### Stage 11: Memory Retrieval
**Source**: `pkg/extproc/req_filter_memory.go`

```
Input: RequestContext
Process:
  1. Query agentic memory (Milvus) for cross-session context
  2. Inject relevant memories into messages
Output: RequestContext with memory context
```

### Stage 12: Context Compression
**Source**: `pkg/extproc/req_filter_compress.go` (ClawRouter-influenced)

```
Input: RequestContext (after RAG + memory injection)
Process:
  1. Compute request body size
  2. If below threshold (default 180KB): continue unchanged
  3. If above threshold, apply compression strategies:
     a. Whitespace normalization (collapse runs of whitespace)
     b. Content deduplication (remove repeated blocks)
     c. JSON compaction (minify JSON in code blocks)
  4. Re-serialize compressed request
Output: Compressed RequestContext
```

**Why this comes after RAG/memory**: RAG and memory injection add the most redundant content (retrieved chunks may overlap, memories may repeat information already in the conversation). Compressing after injection maximizes savings.

### Stage 13: Model Selection & Routing
**Source**: `pkg/extproc/req_filter_route.go`

```
Input: RequestContext + selected Decision
Process:
  1. Session Pinning Check (ClawRouter-influenced):
     a. Compute conversation fingerprint
     b. Check session store for existing pinning
     c. If pinned: use pinned model, skip selection
  2. If not pinned, run Model Selection:
     a. Use Decision's configured algorithm (Elo, RL, AutoMix, etc.)
     b. Apply context-aware fallback chain filtering
     c. Apply degradation policy if active
  3. Set target endpoint via x-vsr-destination-endpoint header
  4. Transform request if needed (Anthropic format, LoRA adapter name)
  5. Inject auth headers from credential chain
Output: Routed request with target endpoint and auth headers
```

---

## 5. Response Pipeline (Detailed)

### Stage R1: Streaming Accumulation
Accumulate streaming SSE chunks into complete response for processing.

### Stage R2: Latency Metrics
Record TTFT (Time to First Token) and TPOT (Time per Output Token).

### Stage R3: Dedup Cache Update (ClawRouter-influenced)
Update dedup cache: move entry from "pending" to "completed" with response.

### Stage R4: Session Tracker Update (ClawRouter-influenced)
Record the model used for this conversation for future session pinning.

### Stage R5: Hallucination Detection
Run NLI-based fact-checking against source documents (if enabled for this Decision).

### Stage R6: Cost Tracking
Compute cost based on token usage and model pricing. Update per-decision and per-user budgets.

### Stage R7: Cache Population
Store response in semantic cache with Decision-specific TTL (if cache plugin enabled).

### Stage R8: Memory Extraction
Asynchronously extract key information from the response for agentic memory storage.

### Stage R9: Rate Limit Reporting
Report token usage to rate limiters for quota tracking.

---

## 6. Fast-Path Pre-Classifier

This is the primary ClawRouter-influenced component. It provides a fast heuristic classification that can bypass the full neural classification pipeline.

### Scoring Dimensions

```go
type FastPathWeights struct {
    TokenCount          float64 `yaml:"tokenCount"          default:"0.08"`
    CodePresence        float64 `yaml:"codePresence"        default:"0.15"`
    ReasoningMarkers    float64 `yaml:"reasoningMarkers"    default:"0.18"`
    TechnicalTerms      float64 `yaml:"technicalTerms"      default:"0.10"`
    CreativeMarkers     float64 `yaml:"creativeMarkers"     default:"0.05"`
    SimpleIndicators    float64 `yaml:"simpleIndicators"    default:"0.02"`
    MultiStepPatterns   float64 `yaml:"multiStepPatterns"   default:"0.12"`
    QuestionComplexity  float64 `yaml:"questionComplexity"  default:"0.05"`
    ImperativeVerbs     float64 `yaml:"imperativeVerbs"     default:"0.03"`
    ConstraintIndicators float64 `yaml:"constraintIndicators" default:"0.04"`
    OutputFormat        float64 `yaml:"outputFormat"        default:"0.03"`
    ReferenceComplexity float64 `yaml:"referenceComplexity" default:"0.02"`
    NegationComplexity  float64 `yaml:"negationComplexity"  default:"0.01"`
    DomainSpecificity   float64 `yaml:"domainSpecificity"   default:"0.02"`
    AgenticTask         float64 `yaml:"agenticTask"         default:"0.04"`
}
```

### Classification Flow

```
                    ┌─────────────────┐
                    │  Request Body   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Extract Text   │
                    │  (user + system │
                    │   messages)     │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
      ┌──────────────┐ ┌──────────┐ ┌──────────────┐
      │ Dimension 1  │ │ Dim 2..13│ │ Dimension 14 │  ... (parallel)
      │ Token Count  │ │   ...    │ │ Domain Spec. │
      │ score: [-1,1]│ │          │ │ score: [-1,1]│
      └──────┬───────┘ └────┬─────┘ └──────┬───────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                    ┌────────▼────────┐
                    │  Weighted Sum   │
                    │  Σ(wᵢ × sᵢ)    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Override Check  │
                    │ • 2+ reasoning  │
                    │   keywords →    │
                    │   REASONING     │
                    │ • tools[] →     │
                    │   AGENTIC       │
                    │ • large context │
                    │   → COMPLEX     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Tier Mapping   │
                    │  < 0.0 → SIMPLE │
                    │  0.0-0.3 → MED  │
                    │  0.3-0.5 → CPLX │
                    │  > 0.5 → REASON │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Confidence     │
                    │  σ(|score -     │
                    │   boundary|)    │
                    └────────┬────────┘
                             │
                ┌────────────┼────────────┐
                ▼                         ▼
        conf >= 0.7                conf < 0.7
    ┌──────────────┐          ┌──────────────┐
    │ Map tier to  │          │ Mark as      │
    │ Decision →   │          │ "ambiguous"  │
    │ Skip neural  │          │ → Run full   │
    │ classifiers  │          │ classification│
    └──────────────┘          └──────────────┘
```

### Dimension Details

Each dimension scorer receives the full message text and returns a score in `[-1.0, 1.0]`:

| Dimension | Positive Signal | Negative Signal | Implementation |
|-----------|----------------|-----------------|----------------|
| Token Count | Long input (>2000 tokens) | Short input (<100 tokens) | `len(tokens)` normalized |
| Code Presence | `function`, `class`, `import`, `async`, `def` | No code markers | Keyword count, normalized |
| Reasoning Markers | `prove`, `theorem`, `derive`, `step by step` | Simple factual patterns | Keyword match (user prompt only) |
| Technical Terms | `algorithm`, `optimize`, `kubernetes`, `distributed` | No technical language | Keyword match + density |
| Creative Markers | `story`, `poem`, `brainstorm`, `imagine` | No creative language | Keyword match |
| Simple Indicators | — | `what is`, `define`, `translate`, `hello` | Keyword match (negative score) |
| Multi-Step | `first...then`, `step 1`, `next...after that` | Single-step instruction | Regex pattern matching |
| Question Complexity | Multiple question marks, nested questions | Single or no questions | `count(?)` + nesting analysis |
| Imperative Verbs | `implement`, `design`, `analyze`, `build` | Simple verbs | Keyword match |
| Constraints | `must`, `ensure`, `constraint`, `within N` | No constraints | Keyword + regex match |
| Output Format | `json`, `yaml`, `csv`, `table`, `markdown` | Free-form output | Keyword match |
| References | `above`, `previous`, `as mentioned`, cross-refs | Self-contained | Pattern matching |
| Negation | `not`, `without`, `except`, `don't` | No negation | Negation density |
| Domain Specificity | Specialized terms (medical, legal, scientific) | General language | Domain dictionary match |

---

## 7. Signal Classification System

The full signal extraction system runs when the fast-path classifier returns low confidence. All signal extractors execute concurrently via goroutines.

```go
type SignalMatches struct {
    Keywords    []KeywordMatch    // Regex, BM25, n-gram matches
    Embeddings  []EmbeddingMatch  // Vector similarity scores
    Domains     []DomainMatch     // Neural category classifications
    Complexity  ComplexityResult  // Multi-factor complexity assessment
    Agentic     AgenticResult     // Agentic task detection
    Language    LanguageResult    // Input language identification
    Modality    ModalityResult    // AR vs. Diffusion detection
    FactCheck   FactCheckResult   // Fact-check necessity score
    Feedback    FeedbackResult    // User satisfaction signals
    AuthZ       AuthZResult       // Authorization evaluation
}
```

### Neural Classifiers (via Candle/Rust)

The router runs BERT-family models natively through Rust bindings, eliminating Python from the production path:

- **Domain Classifier**: BERT or ModernBERT fine-tuned on category taxonomy
- **Jailbreak Detector**: PromptGuard (BERT-based adversarial input detection)
- **PII Detector**: Token classification model for entity types
- **Embedding Generator**: Qwen3, Gemma, mmBERT, or standard BERT for vector representations

Build-tag gating ensures graceful fallback:
- `//go:build cgo_candle` → Full Candle inference
- `//go:build cgo_onnx` → ONNX Runtime inference
- Default (no tags) → Stub implementations that skip neural classification

---

## 8. Decision Engine

The decision engine evaluates boolean expression trees over signal matches.

### Expression Tree Structure

```yaml
decisions:
  - name: "code-review"
    rules:
      type: "AND"
      conditions:
        - signal: "domain"
          category: "coding"
          minConfidence: 0.8
        - type: "OR"
          conditions:
            - signal: "complexity"
              level: "high"
            - signal: "keyword"
              pattern: "review|refactor|optimize"
        - type: "NOT"
          condition:
            signal: "agentic"
            detected: true
```

### Evaluation Algorithm

```
evalNode(node, signals) → (matched: bool, confidence: float64)
  switch node.type:
    case "AND":  return evalAND(node.conditions, signals)
    case "OR":   return evalOR(node.conditions, signals)
    case "NOT":  return evalNOT(node.condition, signals)
    case signal: return matchSignal(node, signals)

evalAND(conditions, signals):
  allMatch = true
  minConfidence = 1.0
  for each condition:
    (matched, conf) = evalNode(condition, signals)
    if !matched: return (false, 0)
    minConfidence = min(minConfidence, conf)
  return (allMatch, minConfidence)

evalOR(conditions, signals):
  bestConfidence = 0
  for each condition:
    (matched, conf) = evalNode(condition, signals)
    if matched: bestConfidence = max(bestConfidence, conf)
  return (bestConfidence > 0, bestConfidence)
```

---

## 9. Model Selection

Once a Decision is selected, the router determines which specific model instance to use.

### Algorithm Registry

```go
type Selector interface {
    Select(ctx context.Context, req *Request, candidates []Model) (*Model, error)
    Method() string
    UpdateFeedback(feedback *Feedback) error
}
```

| Algorithm | Category | Strengths | Best For |
|-----------|----------|-----------|----------|
| Static | Rule-based | Predictable, zero overhead | Fixed model assignments |
| Elo | Statistical | Self-improving, handles preference | Quality-focused selection |
| RouterDC | Neural | Learns from contrastive pairs | High-volume optimization |
| AutoMix | Optimization | Cost-quality Pareto frontier | Budget-constrained |
| Hybrid | Ensemble | Combines multiple signals | Balanced workloads |
| KNN/KMeans/SVM/MLP | ML | Trained on historical data | Pattern-based routing |
| RL-Driven | RL | Thompson Sampling, personalizes | Exploration/exploitation |
| GMTRouter | Graph | User-preference graphs | Personalized routing |
| Latency-Aware | Metrics | TPOT/TTFT optimization | Latency-sensitive apps |

### Context-Aware Fallback (ClawRouter-influenced)

When the primary model fails (429, 5xx, timeout), the fallback chain activates:

```
Primary Model → Fallback 1 → Fallback 2 → ... → Degradation Model
                     │              │
                     ▼              ▼
              [Context window   [Context window
               >= request size   >= request size
               + 10% buffer?]    + 10% buffer?]
               If no: skip       If no: skip
```

Models are filtered by whether they can actually handle the request's context size before attempting fallback.

---

## 10. Session Pinning

**Source**: `pkg/session/` (ClawRouter-influenced)

### State Machine

```
┌───────────────┐    First request    ┌────────────────┐
│   No Session  │───────────────────►│ Pinned (model, │
│               │                    │  endpoint,      │
└───────────────┘                    │  provider)      │
                                     └───────┬────────┘
                                             │
                              ┌──────────────┼──────────────┐
                              ▼              ▼              ▼
                        Subsequent     TTL Expires     Model becomes
                        requests       (configurable)  unavailable
                        use pinned     ┌──────────┐   ┌──────────┐
                        model          │ Session  │   │ Re-select│
                                       │ Expired  │   │ + re-pin │
                                       └──────────┘   └──────────┘
```

### Conversation Fingerprinting

Sessions are identified by a fingerprint derived from:
1. First message content hash (or conversation ID if provided)
2. Client IP (optional)
3. API key hash (optional)
4. Custom header values (configurable)

### Storage Backends

- **In-memory**: LRU cache with TTL eviction (default)
- **Redis**: Distributed session storage for multi-instance deployments

---

## 11. Request Deduplication

**Source**: `pkg/dedup/` (ClawRouter-influenced)

### Algorithm

```
┌──────────────┐
│   Request    │
└──────┬───────┘
       │
┌──────▼───────┐     ┌─────────────────────────────────────┐
│ Canonicalize │     │  Canonicalization steps:             │
│   JSON       │────►│  1. Parse JSON                      │
│              │     │  2. Sort all keys recursively        │
│              │     │  3. Strip: timestamp, request_id,    │
│              │     │     trace headers                    │
│              │     │  4. Re-serialize deterministically   │
└──────┬───────┘     └─────────────────────────────────────┘
       │
┌──────▼───────┐
│  SHA-256     │
│   Hash       │
└──────┬───────┘
       │
┌──────▼───────┐     ┌─────────────┐
│ LRU Cache    │────►│ Hit: return │────► Cached response
│  Lookup      │     │ cached resp │      (short-circuit)
│              │     └─────────────┘
│              │     ┌─────────────┐
│              │────►│ Pending:    │────► Wait for in-flight
│              │     │ same hash   │      request to complete
│              │     │ in progress │
│              │     └─────────────┘
│              │     ┌─────────────┐
│              │────►│ Miss:       │────► Mark pending,
│              │     │ new request │      continue pipeline
└──────────────┘     └─────────────┘
```

### Configuration

```yaml
deduplication:
  enabled: true
  windowSeconds: 30        # TTL for cache entries
  maxEntries: 10000        # LRU eviction limit
  hashAlgorithm: "sha256"  # Hash function
  stripFields:             # Fields to remove before hashing
    - "timestamp"
    - "request_id"
    - "x-request-id"
```

---

## 12. Context Compression

**Source**: `pkg/compression/` (ClawRouter-influenced)

### Strategies

| Strategy | Description | Savings |
|----------|-------------|---------|
| `whitespace` | Collapse runs of whitespace, normalize line endings | 5-15% |
| `dedup` | Detect and remove duplicate text blocks (common in RAG) | 10-40% |
| `json-compact` | Minify JSON/YAML in code blocks and tool responses | 10-30% |

Strategies are applied in sequence and are individually toggleable per Decision.

### Application Point

Compression runs **after** RAG injection and memory retrieval (Stage 12 in the request pipeline). This is deliberate — RAG and memory add the most redundant content, so compressing after injection maximizes savings.

---

## 13. Agentic Task Detection

**Source**: `pkg/classification/agentic_classifier.go` (ClawRouter-influenced)

### Detection Signals

| Signal | Indicators |
|--------|-----------|
| Tool use | `tools[]` array present in request |
| File operations | Keywords: `read_file`, `write_file`, `create`, `edit`, `delete` |
| Shell operations | Keywords: `run_command`, `execute`, `bash`, `terminal` |
| Iterative patterns | Keywords: `try again`, `retry`, `fix`, `debug`, `test and fix` |
| Multi-step planning | Keywords: `step 1`, `first...then`, `plan`, `implement` |
| Code modification | Presence of diff-like content, file paths, line numbers |

### Output

```go
type AgenticResult struct {
    Detected   bool    // Is this an agentic task?
    Score      float64 // Confidence score [0, 1]
    TaskType   string  // "coding", "research", "automation", "general"
    MultiStep  bool    // Does this involve multiple sequential steps?
}
```

This signal feeds into the decision engine like any other signal type, and can also influence model selection (e.g., preferring models with strong tool-use capabilities).

---

## 14. Graceful Degradation

**Source**: `pkg/selection/degradation.go` (ClawRouter-influenced)

### Trigger Conditions

| Trigger | Source | Behavior |
|---------|--------|----------|
| Rate limit exceeded | Rate limiter | Route to fallback model pool |
| Budget exhausted | Cost tracker | Route to cheapest available model |
| Primary model unavailable | Health check / 5xx | Walk fallback chain |
| All fallbacks exhausted | Fallback chain | Route to degradation model |

### Configuration

```yaml
decisions:
  - name: "general"
    plugins:
      degradationPolicy:
        enabled: true
        triggers:
          - "rate_limit"
          - "budget_exceeded"
          - "provider_error"
        fallbackModel: "nvidia/gpt-oss-120b"  # Free tier
        notifyHeaders:
          X-Degraded: "true"
          X-Original-Model: "{{requested_model}}"
```

When degradation activates, custom headers inform the client that a lower-tier model was used.

---

## 15. Security Pipeline

Three sequential stages, each independently configurable per Decision:

### Jailbreak Detection
- **Model**: PromptGuard (BERT-based) via Candle or vLLM backend
- **Action**: Block (return 400) if score exceeds threshold
- **Latency**: ~5ms via Candle, ~20ms via vLLM

### PII Detection
- **Model**: Token classification (NER) via Candle
- **Actions**: Block, redact, or log (per-Decision policy)
- **Entity types**: Names, emails, phone numbers, SSNs, credit cards, addresses
- **Latency**: ~8ms via Candle

### Hallucination Mitigation
- **Model**: NLI-based fact-checking (response phase)
- **Sources**: RAG-provided documents, conversation history
- **Action**: Flag, append warning, or block

---

## 16. Caching Architecture

### Semantic Cache

Unlike traditional exact-match caching, the semantic cache uses vector similarity:

```
Request → Embedding → HNSW Index Query → Similar cached response?
                                              │
                              ┌────────────────┼────────────────┐
                              ▼                ▼                ▼
                        similarity         similarity      similarity
                        >= 0.95            0.85 - 0.95     < 0.85
                        ┌──────────┐   ┌──────────────┐   ┌──────────┐
                        │  Cache   │   │  Return with │   │  Cache   │
                        │   Hit    │   │  "similar"   │   │   Miss   │
                        │          │   │   flag       │   │          │
                        └──────────┘   └──────────────┘   └──────────┘
```

### Storage Backends

| Backend | Use Case | Pros | Cons |
|---------|----------|------|------|
| In-memory | Development, single-instance | Zero latency, simple | Not distributed, lost on restart |
| Redis | Multi-instance, persistent | Distributed, fast | Requires Redis deployment |
| Milvus | Large-scale, vector-native | Purpose-built for vectors, scalable | Heavier infrastructure |

### HNSW Index

The router includes a custom HNSW (Hierarchical Navigable Small World) implementation with:
- AMD64 SIMD assembly for distance computation
- Configurable M (connections per layer) and efConstruction parameters
- Thread-safe concurrent reads with periodic background rebuilds

---

## 17. Multi-Model Execution

The `looper/` package enables sending a single request to multiple models:

| Strategy | Flow | Use Case |
|----------|------|----------|
| **Confidence** | Model₁ → check confidence → if low → Model₂ → ... | Quality escalation |
| **Ratings** | Model₁ ∥ Model₂ ∥ Model₃ → rate → select best | Maximum quality |
| **ReMoM** | Round 1: all models → Round 2: refine with best → ... | Iterative refinement |
| **RL-Driven** | RL agent selects sequence based on learned policy | Adaptive sequencing |

---

## 18. Memory System

The agentic memory system stores cross-session context in Milvus:

```
Response → Memory Extractor → Key facts, actions, decisions
                                        │
                                        ▼
                               Milvus (vector store)
                                        │
            New Request → Memory Query ──┘
                              │
                              ▼
                    Inject relevant memories
                    into request context
```

Features:
- Deduplication (avoid storing repeated memories)
- Scoring (relevance decay over time)
- Per-user and per-session scoping

---

## 19. Configuration Architecture

### Source Priority

```
Kubernetes CRDs (highest priority)
        │
        ▼
YAML Configuration Files (with hot-reload via fsnotify)
        │
        ▼
Environment Variables (lowest priority)
```

### Hot-Reload Mechanism

```go
// Atomic pointer swap for zero-downtime updates
type RouterService struct {
    router atomic.Pointer[Router]  // Lock-free reads
}

// Watcher goroutine
func (s *RouterService) watchConfig() {
    for event := range watcher.Events {
        newConfig := loadConfig(event.Name)
        newRouter := buildRouter(newConfig)
        s.router.Store(newRouter)  // Atomic swap
    }
}

// Request handler — lock-free
func (s *RouterService) HandleRequest(ctx context.Context, req *Request) {
    router := s.router.Load()  // Atomic load, no lock
    router.Process(ctx, req)
}
```

---

## 20. Kubernetes Integration

### Custom Resource Definitions

| CRD | Purpose |
|-----|---------|
| `IntelligentPool` | Defines a pool of model backends with selection configuration |
| `IntelligentRoute` | Defines routing rules, decisions, and pipeline configuration |
| `SemanticRouter` | Top-level resource managed by the Kubernetes operator |

### Operator

The Kubebuilder-based operator watches for CRD changes and reconciles the router configuration:

```
CRD Change Event → Operator Reconcile → Generate Config → Hot-Reload Router
```

---

## 21. Observability

### Metrics (Prometheus)

| Metric | Type | Description |
|--------|------|-------------|
| `sr_request_total` | Counter | Total requests by decision, model, status |
| `sr_request_duration_seconds` | Histogram | End-to-end request latency |
| `sr_ttft_seconds` | Histogram | Time to first token |
| `sr_tpot_seconds` | Histogram | Time per output token |
| `sr_fastpath_decisions_total` | Counter | Fast-path classifications by tier + confidence |
| `sr_fastpath_bypassed_total` | Counter | Requests that skipped neural classification |
| `sr_dedup_hits_total` | Counter | Deduplicated requests |
| `sr_session_pins_total` | Counter | Session pinning events |
| `sr_compression_bytes_saved` | Counter | Bytes saved by context compression |
| `sr_degradation_total` | Counter | Degradation activations by trigger |
| `sr_cache_hits_total` | Counter | Semantic cache hits |
| `sr_security_blocks_total` | Counter | Security pipeline blocks by type |
| `sr_cost_total` | Counter | Estimated cost by model and decision |

### Tracing (OpenTelemetry)

Distributed traces span the full request lifecycle with spans for:
- Each pipeline filter stage
- Neural classifier inference
- Cache lookups
- Backend model calls
- Response processing

### Logging

Structured JSON logging with:
- Request/response metadata
- Decision selection reasoning
- Security findings
- Performance timings

---

## 22. Data Flow Diagrams

### Happy Path (Fast-Path Hit)

```
Client → Envoy → ExtProc Header Phase → ExtProc Body Phase
                                              │
                                         1. Dedup: miss
                                         2. FastPath: SIMPLE (conf=0.92)
                                              │ [skip neural classification]
                                         3. Jailbreak: clean
                                         4. PII: clean
                                         5. Cache: miss
                                         6. Route: model-x at endpoint-y
                                              │
                                         Envoy → Backend → Response
                                              │
                                         Response Phase:
                                         1. Metrics recorded
                                         2. Session pinned
                                         3. Cache populated
                                         4. Dedup updated
```

**Total router overhead: ~2-5ms**

### Full Classification Path

```
Client → Envoy → ExtProc Body Phase
                      │
                 1. Dedup: miss
                 2. FastPath: ambiguous (conf=0.45)
                 3. Signal Extraction (parallel, ~20-50ms):
                    ├── Keywords: matched "kubernetes", "deploy"
                    ├── Embedding: 0.87 similarity to "devops" cluster
                    ├── Domain: "infrastructure" (0.91 confidence)
                    ├── Complexity: "high"
                    ├── Agentic: false
                    └── Language: "en"
                 4. Decision Engine: "infra-complex" (0.89)
                 5. Jailbreak: clean
                 6. PII: clean
                 7. Cache: miss
                 8. RAG: 3 chunks injected
                 9. Compression: 210KB → 165KB (21% savings)
                10. Session: new pin → claude-sonnet-4
                11. Route: claude-sonnet-4 at anthropic-api
                      │
                 Envoy → Backend → Response
```

**Total router overhead: ~30-70ms** (dominated by neural classification)

---

## 23. Module Dependency Map

```
extproc (pipeline orchestration)
  ├── dedup          (request deduplication)
  ├── fastpath       (fast-path pre-classifier)
  │   └── config     (weights and thresholds)
  ├── classification (signal extraction)
  │   ├── candle-binding (Rust ML inference)
  │   └── agentic_classifier
  ├── decision       (expression tree evaluator)
  ├── selection      (model selection algorithms)
  │   └── degradation (fallback policy)
  ├── session        (conversation pinning)
  ├── cache          (semantic caching)
  │   └── hnsw       (vector index)
  ├── compression    (context compression)
  ├── memory         (agentic memory)
  ├── mcp            (tool integration)
  ├── looper         (multi-model execution)
  ├── ratelimit      (quota enforcement)
  ├── authz          (credential management)
  ├── observability  (metrics, traces, logs)
  └── config         (configuration model)
      └── k8s        (CRD controller)
```

---

## 24. Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Fast-path classification latency | <1ms p99 | Pure heuristic, no I/O |
| Full classification latency | <50ms p99 | Parallel neural inference via Candle |
| Dedup lookup latency | <0.1ms p99 | In-memory LRU hash lookup |
| Session pinning lookup | <0.5ms p99 | In-memory or Redis |
| Cache lookup latency | <5ms p99 | HNSW similarity search |
| Context compression | <2ms p99 | String processing only |
| Total router overhead (fast path) | <5ms p99 | Dedup + FastPath + security + routing |
| Total router overhead (full path) | <70ms p99 | Full neural classification pipeline |
| Config hot-reload | <100ms | Atomic pointer swap |
| Memory footprint | <2GB | With loaded ML models |
