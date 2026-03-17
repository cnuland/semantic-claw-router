# Claude Code Instructions: Semantic Claw Router

You are building and maintaining **Semantic Claw Router** — an intelligent LLM request routing layer implemented in Python. It merges the architecture of [vllm-project/semantic-router](https://github.com/vllm-project/semantic-router) with routing intelligence from [BlockRunAI/ClawRouter](https://github.com/BlockRunAI/ClawRouter).

## Project Identity

- **Language**: Python 3.10+
- **Primary lineage**: vLLM Semantic Router (pipeline architecture, decision engine, provider abstraction, observability)
- **Influenced by**: ClawRouter (fast-path classifier, session pinning, dedup, compression, agentic detection, graceful degradation)
- **License**: Apache 2.0 (ClawRouter-influenced concepts noted in source headers)
- **API**: OpenAI-compatible `/v1/chat/completions`

## Hard Constraints

1. **Python is the language** — this is a pure Python project. No Go, Rust, or native dependencies.
2. **OpenAI-compatible API** — the router exposes `/v1/chat/completions`. Any OpenAI SDK client (OpenClaw, Cursor, LM Studio, etc.) works unmodified.
3. **All features are configurable** — every pipeline stage (dedup, session, compression, degradation) can be enabled/disabled via YAML config.
4. **No hardcoded secrets** — API keys and endpoints use `${VAR}` environment variable expansion in YAML. Never commit secrets.
5. **Maintain the 9-stage pipeline** — new functionality is added as new pipeline stages or new classifier dimensions. Do not break the sequential flow.
6. **Signal-based architecture** — new classification capabilities are added as scoring dimensions in `router/fastpath.py` with configurable weights.
7. **Provider abstraction** — new LLM providers implement the `LLMProvider` abstract class in `providers/base.py`.
8. **Keep dependencies minimal** — core routing requires only `aiohttp`, `pyyaml`, and the standard library. Provider-specific deps (like `google-generativeai`) are optional.

## Architecture Reference

Read `docs/pipeline.md` for the full 9-stage pipeline and 15-dimension classifier design. Key files:

| Area | Primary Files |
|------|---------------|
| HTTP server + pipeline | `src/semantic_claw_router/server.py` |
| CLI entrypoint | `src/semantic_claw_router/cli.py` |
| YAML config + dataclasses | `src/semantic_claw_router/config.py` |
| Core types (tiers, backends) | `src/semantic_claw_router/router/types.py` |
| 15-dim fast-path classifier | `src/semantic_claw_router/router/fastpath.py` |
| Semantic embedding classifier | `src/semantic_claw_router/router/semantic.py` |
| Decision engine (tier → model) | `src/semantic_claw_router/router/decision.py` |
| Request deduplication | `src/semantic_claw_router/pipeline/dedup.py` |
| Session pinning | `src/semantic_claw_router/pipeline/session.py` |
| Context compression | `src/semantic_claw_router/pipeline/compress.py` |
| vLLM / OpenAI provider | `src/semantic_claw_router/providers/vllm.py` |
| Google Gemini provider | `src/semantic_claw_router/providers/gemini.py` |
| Abstract provider interface | `src/semantic_claw_router/providers/base.py` |
| Structured logging | `src/semantic_claw_router/observability/logging.py` |
| Prometheus metrics | `src/semantic_claw_router/observability/metrics.py` |

## Implementation Guidance

### Adding a new scoring dimension
1. Add the dimension name and default weight to the `FastPathConfig` dataclass in `config.py`
2. Implement the scoring function in `router/fastpath.py` (returns -1.0 to +1.0)
3. Add it to the dimension evaluation loop in `FastPathClassifier.classify()`
4. Update the config YAML schema in `docs/configuration.md`
5. Write unit tests in `tests/test_fastpath.py`

### Adding a new LLM provider
1. Create `src/semantic_claw_router/providers/<name>.py`
2. Implement `LLMProvider` abstract class (`chat_completion()`, `health_check()`)
3. Register the provider name in `server.py` provider factory
4. Add any required dependencies to `pyproject.toml` as optional extras
5. Write integration tests in `tests/test_integration.py`

### Adding a new pipeline stage
1. Implement the stage logic as a method on the server or as a new module in `pipeline/`
2. Insert it at the correct position in the 9-stage pipeline in `server.py`
3. Add configuration to `config.py` dataclasses
4. Write unit tests
5. Update `docs/pipeline.md`

### ClawRouter-influenced modules
When implementing features inspired by ClawRouter, include this header comment:

```python
# This module implements [feature description]
# inspired by ClawRouter (https://github.com/BlockRunAI/ClawRouter).
# Original concept by BlockRun under MIT License.
# Re-implemented in Python for the Semantic Claw Router pipeline.
```

## Local Development (test before deploying to OpenShift)

Always test router changes locally before pushing to the OpenShift BuildConfig. The cluster build/deploy cycle takes ~20 minutes.

```bash
# 1. Source API keys
export $(cat .env | xargs)

# 2. Port-forward the cluster Ollama so the local model backend works
oc port-forward svc/ollama-qwen3-30b 11434:11434 -n gpt-oss &

# 3. Run the router locally
python -m semantic_claw_router -c config.local.yaml

# 4. Run the test harness (validates streaming, tool calls, routing)
python scripts/test_local.py

# 5. Test individual scenarios
python scripts/test_local.py --test simple    # Should route to local Qwen3
python scripts/test_local.py --test complex   # Should route to Gemini
python scripts/test_local.py --test tools     # Tool call history (Gemini regression test)
python scripts/test_local.py --test no-name   # Tool result without name field

# 6. When satisfied, deploy to OpenShift
oc start-build semantic-claw-router --from-dir=. -n gpt-oss --follow
```

Key files for local dev:
- `.env` — API keys (gitignored)
- `config.local.yaml` — local config pointing to localhost:11434 (gitignored)
- `scripts/test_local.py` — test harness mimicking OpenClaw's request patterns

## Testing

```bash
# All unit tests (80 tests, < 1 second)
pytest tests/ -v

# Integration tests (requires VLLM_ENDPOINT and GEMINI_API_KEY env vars)
pytest tests/test_integration.py -v

# Linter
ruff check src/ tests/

# Run the 21-prompt example
python examples/run_example.py
```

- All new modules must have corresponding `test_*.py` files with >80% coverage
- Integration tests go in `tests/test_integration.py` and are skipped when env vars are missing
- Benchmark tests for hot-path code (classifier must prove <1ms p99 latency)

## Code Style

- Type hints on public APIs
- Dataclasses for configuration and structured types
- `async/await` for all I/O operations
- Structured logging via the observability module
- Metrics via the Prometheus-compatible metrics collector

## File Guidance

- `docs/pipeline.md` — 9-stage pipeline and 15-dimension classifier deep dive
- `docs/getting-started.md` — Installation, configuration, running the router
- `docs/configuration.md` — Full YAML config reference
- `docs/api-reference.md` — HTTP endpoints, headers, session pinning
- `docs/benchmarks.md` — Performance data and test coverage
- `docs/contributing.md` — Development setup and contribution areas
- `docs/architecture.md` — Full system architecture specification
- `docs/research.md` — Research foundations and project analysis

## Non-Goals

- Replacing the Python implementation with Go/Rust
- Adding blockchain/payment protocols
- Forcing any single LLM provider as a hard dependency
- Breaking OpenAI API compatibility
- Hardcoding model catalogs (use YAML config)
