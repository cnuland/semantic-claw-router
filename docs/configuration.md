# Configuration Reference

The router is configured via YAML with `${VAR}` environment variable expansion. See [examples/config.yaml](../examples/config.yaml) for a complete example.

---

## Models

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

## Tier-to-Model Mapping

```yaml
default_tier_models:
  SIMPLE: "local-model"           # Free, fast
  MEDIUM: "local-model"
  COMPLEX: "cloud-model"          # Capable, paid
  REASONING: "cloud-model"
```

## Fast-Path Classifier

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

## Request Deduplication

```yaml
dedup:
  enabled: true
  window_seconds: 30              # Cache TTL
  max_entries: 10000              # LRU capacity
```

## Session Pinning

```yaml
session:
  enabled: true
  ttl_seconds: 3600               # 1 hour session lifetime
  max_sessions: 10000
```

## Context Compression

```yaml
compression:
  enabled: true
  threshold_bytes: 184320         # 180 KB trigger
  strategies:
    - whitespace                  # Collapse runs of whitespace
    - dedup                       # Remove duplicate paragraphs
    - json_compact                # Minify JSON in code fences
```

## Graceful Degradation

```yaml
degradation:
  enabled: true
  fallback_model: "local-model"
  triggers:
    - provider_error              # 5xx responses
    - rate_limit                  # 429 responses
    - timeout                     # Request timeout
```

## Observability

```yaml
observability:
  log_level: "INFO"               # DEBUG, INFO, WARNING, ERROR
  log_format: "json"              # "json" or "text"
  metrics_enabled: true
```

## Environment Variable Expansion

All string values in the YAML support environment variable expansion:

- `${VAR}` — replaced with the value of `VAR`, or kept as-is if unset
- `${VAR:-default}` — replaced with the value of `VAR`, or `default` if unset
