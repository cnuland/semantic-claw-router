# Claude Code Instructions: Semantic Router

You are building and maintaining an **intelligent LLM request routing layer** that combines the architecture of [vllm-project/semantic-router](https://github.com/vllm-project/semantic-router) with concepts from [BlockRunAI/ClawRouter](https://github.com/BlockRunAI/ClawRouter).

## Project Identity

- **Primary lineage**: vllm-project/semantic-router (Go/Rust/Python, Envoy ExtProc, Kubernetes-native)
- **Influenced by**: BlockRunAI/ClawRouter (tiered classification, session pinning, dedup, compression, agentic detection, graceful degradation)
- **License**: Apache 2.0 (ClawRouter-influenced modules note MIT attribution in source headers)

## Hard Constraints

1. **Go is the primary language** for all router core code. Rust for ML inference (Candle bindings). Python only for training pipelines and CLI tooling.
2. **Envoy ExtProc is the integration model** — the router operates as a gRPC external processor. Do not introduce alternative proxy models.
3. **All ClawRouter-influenced features are optional** — they must be disabled by default and enabled via configuration. Existing behavior must not change.
4. **No blockchain/x402 dependencies** — the x402 payment protocol from ClawRouter is explicitly out of scope for the core. It may only exist as a future plugin.
5. **Maintain the filter pipeline pattern** — new functionality is added as `req_filter_*.go` or `res_filter_*.go` files in `pkg/extproc/`. Do not break the sequential pipeline.
6. **Signal-based architecture** — new classification capabilities are added as signal types in `pkg/classification/`. They feed into the existing decision engine.
7. **Factory + Registry pattern for extensibility** — new selection algorithms, cache backends, and classifiers use factories and registries for runtime switching.
8. **Build tags for optional native dependencies** — features requiring Candle, ONNX, or other native libraries use Go build tags with stub fallbacks.
9. **Keep dependencies minimal** — the Go router should have zero Python runtime dependency in production. Candle/Rust handles all ML inference natively.
10. **Kubernetes CRDs are first-class** — any new configuration surface must have corresponding CRD schema updates.

## Architecture Reference

Read `architecture.md` for the full system design. Key files:

| Area | Primary Files |
|------|---------------|
| Request pipeline | `pkg/extproc/processor_req_body.go`, `pkg/extproc/req_filter_*.go` |
| Response pipeline | `pkg/extproc/processor_res_body.go`, `pkg/extproc/res_filter_*.go` |
| Decision engine | `pkg/decision/` |
| Signal classifiers | `pkg/classification/` |
| Model selection | `pkg/selection/` |
| Fast-path classifier | `pkg/fastpath/` |
| Session pinning | `pkg/session/` |
| Request deduplication | `pkg/dedup/` |
| Context compression | `pkg/compression/` |
| Agentic detection | `pkg/classification/agentic_classifier.go` |
| Configuration | `pkg/config/config.go` |
| Candle ML bindings | `candle-binding/` |

## Implementation Guidance

### Adding a new request filter
1. Create `pkg/extproc/req_filter_<name>.go`
2. Implement the filter function with signature matching existing filters
3. Add the filter to the pipeline in `processor_req_body.go` at the correct position
4. Add configuration to `pkg/config/config.go`
5. Add corresponding CRD fields to `pkg/apis/vllm.ai/v1alpha1/`
6. Write unit tests in `pkg/extproc/req_filter_<name>_test.go`

### Adding a new signal type
1. Create `pkg/classification/<signal>_classifier.go`
2. Implement the classifier interface
3. Register it in the parallel signal extraction in `classification/unified_classifier.go`
4. Add signal type to the decision engine's condition matching
5. Add configuration schema
6. Write tests

### Adding a new selection algorithm
1. Create `pkg/selection/<algorithm>.go`
2. Implement the `Selector` interface (`Select()`, `Method()`, `UpdateFeedback()`)
3. Register in `pkg/selection/factory.go`
4. Add to the `Registry` for runtime switching
5. Write tests

### ClawRouter-influenced modules
When implementing features inspired by ClawRouter, include this header comment:

```go
// This module implements [feature description]
// inspired by ClawRouter (https://github.com/BlockRunAI/ClawRouter).
// Original concept by BlockRun under MIT License.
// Re-implemented in Go for the semantic-router ExtProc pipeline.
```

## Testing Requirements

- All new packages must have `_test.go` files with >80% coverage
- Integration tests go in `tests/integration/`
- Benchmark tests for hot-path code (filters, classifiers, selectors)
- Fast-path classifier must include benchmarks proving <1ms p99 latency
- Session pinning must include tests for TTL expiry and concurrent access
- Dedup must include tests for hash collision resistance and LRU eviction

## Code Style

- Follow existing Go conventions in the codebase (gofmt, golint, govet)
- Use `context.Context` propagation through the entire pipeline
- Structured logging via the existing observability package
- Metrics via Prometheus (register new counters/histograms for new features)
- Tracing spans for any operation >1ms

## File Guidance

- `architecture.md` — Full system architecture and pipeline design
- `research.md` — Research foundations and project analysis
- `skills.md` — Capabilities and feature reference
- `README.md` — Project overview and quickstart

## Non-Goals

- Replacing Envoy with a custom proxy
- Adding Python to the production runtime path
- Implementing x402/blockchain payment in core
- Breaking OpenAI API compatibility
- Removing or altering any existing signal types, selection algorithms, or filter stages
- Adding hard dependencies on any single LLM provider
