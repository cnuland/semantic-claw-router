# Contributing

We welcome contributions! The project is designed for easy extension.

---

## Development Setup

```bash
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

## Areas for Contribution

- **New providers** — Add support for Anthropic, Azure, Bedrock, Ollama, etc.
- **New classifier dimensions** — Add scoring dimensions to the fast-path classifier
- **Neural classifiers** — Implement BERT-based classification for borderline cases
- **Semantic caching** — HNSW vector similarity cache for near-duplicate requests
- **Security pipeline** — Jailbreak detection, PII redaction
- **Kubernetes operator** — CRD-based configuration with `IntelligentPool` and `IntelligentRoute` resources
- **Streaming support** — True SSE streaming for Gemini provider (currently buffers)

## Project Structure

```
semantic-claw-router/
├── src/semantic_claw_router/
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
├── tests/                        # 89 tests (80 unit + 9 integration)
├── examples/                     # Example config + 21-prompt runner
└── docs/                         # Documentation
```
