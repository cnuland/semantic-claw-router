# Getting Started

Full guide for installing, configuring, and running the Semantic Claw Router.

---

## Prerequisites

- Python 3.10+
- At least one LLM backend (vLLM, Ollama, OpenAI API, Gemini API, etc.)

## Installation

```bash
git clone https://github.com/cnuland/semantic-claw-router.git
cd semantic-claw-router
pip install -e ".[dev]"
```

## Configuration

Create a configuration file with your model backends. The router supports `${VAR}` and `${VAR:-default}` environment variable expansion in YAML values.

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

### Sourcing Credentials from OpenShift

If you're running models on OpenShift, source credentials from the cluster:

```bash
export GEMINI_API_KEY=$(oc get secret gemini-api-key -n aiops-harness \
    -o jsonpath='{.data.GEMINI_API_KEY}' | base64 -d)
export VLLM_ENDPOINT=$(oc get route qwen3-coder-next -n llm-serving \
    -o jsonpath='https://{.spec.host}')
```

## Running the Router

```bash
export GEMINI_API_KEY="your-api-key-here"
semantic-claw-router --config config.yaml --port 8080
```

## Sending Requests

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
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "What is Python?"}]}'
```

## Using with OpenClaw

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

This is the same integration pattern as [ClawRouter](https://github.com/BlockRunAI/ClawRouter), which was originally built for OpenClaw. The `baseUrl` approach also works with any OpenAI SDK client — LM Studio, Continue, Cursor, or any tool that supports custom API endpoints.

## Running the Example Suite

The example runner sends 21 simulated AI coding assistant prompts across all complexity tiers:

```bash
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
  ...
  Classification accuracy: 18/21 (86%)
```

## Running Tests

```bash
# All unit tests (80 tests, < 1 second)
pytest tests/

# With coverage
pytest tests/ --cov=semantic_claw_router --cov-report=term-missing

# Integration tests (requires live endpoints)
VLLM_ENDPOINT=https://your-host GEMINI_API_KEY=your-key \
    pytest tests/test_integration.py -m integration -v
```
