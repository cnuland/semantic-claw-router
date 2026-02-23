# API Reference

The router exposes an **OpenAI-compatible HTTP API**.

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Route a chat completion request |
| `GET` | `/v1/models` | List configured model backends |
| `GET` | `/health` | Health check |
| `GET` | `/metrics` | Routing metrics summary (JSON) |

## Request Format

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

## Response Headers

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

## Session Pinning

Pass an `x-session-id` header to pin multi-turn conversations to the same model:

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

## Metrics Endpoint

`GET /metrics` returns a JSON summary:

```json
{
  "total_requests": 100,
  "latency": {
    "mean_ms": 312,
    "p50_ms": 220,
    "p99_ms": 1450
  },
  "tier_distribution": {
    "SIMPLE": 40,
    "MEDIUM": 35,
    "COMPLEX": 15,
    "REASONING": 10
  },
  "model_distribution": {
    "local-model": 75,
    "cloud-model": 25
  },
  "tokens": {
    "total_input": 45000,
    "total_output": 22000
  }
}
```
