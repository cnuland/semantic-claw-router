#!/usr/bin/env python3
"""Local test harness for semantic-claw-router.

Sends the same kinds of requests OpenClaw sends (stream: true, tool call
history, SSE expectations) and validates the responses. Run with:

    python scripts/test_local.py [--base-url http://localhost:8080]

Prerequisites:
    1. Router running locally:
       export $(cat .env | xargs)
       python -m semantic_claw_router -c config.local.yaml
    2. Ollama port-forwarded (or local):
       oc port-forward svc/ollama-qwen3-30b 11434:11434 -n gpt-oss
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import httpx

DEFAULT_BASE = "http://localhost:8080"


def test_health(client: httpx.Client, base: str) -> bool:
    """Test /health endpoint."""
    print("\n=== Test: Health Check ===")
    try:
        r = client.get(f"{base}/health")
        print(f"  Status: {r.status_code}")
        print(f"  Body: {r.text[:200]}")
        return r.status_code == 200
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_models(client: httpx.Client, base: str) -> bool:
    """Test /v1/models endpoint."""
    print("\n=== Test: Model List ===")
    try:
        r = client.get(f"{base}/v1/models")
        print(f"  Status: {r.status_code}")
        data = r.json()
        models = [m["id"] for m in data.get("data", [])]
        print(f"  Models: {models}")
        return r.status_code == 200 and len(models) > 0
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_simple_streaming(client: httpx.Client, base: str) -> bool:
    """Test a simple prompt with stream:true — should route to local model."""
    print("\n=== Test: Simple Streaming (should route to local model) ===")
    body = {
        "model": "auto",
        "stream": True,
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
    }
    try:
        collected = _collect_sse(client, base, body)
        if collected is None:
            return False
        print(f"  Full response: {collected['text'][:200]}")
        print(f"  Chunks received: {collected['chunk_count']}")
        print(f"  Has [DONE]: {collected['has_done']}")
        print(f"  Model: {collected['model']}")
        return collected["has_done"] and len(collected["text"]) > 0
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_complex_streaming(client: httpx.Client, base: str) -> bool:
    """Test a complex prompt with stream:true — should route to Gemini."""
    print("\n=== Test: Complex Streaming (should route to Gemini) ===")
    body = {
        "model": "auto",
        "stream": True,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": (
                    "Analyze the trade-offs between microservices and monolithic "
                    "architectures for a startup with 5 engineers. Consider "
                    "operational complexity, deployment velocity, and debugging. "
                    "Provide a structured recommendation with pros and cons."
                ),
            },
        ],
    }
    try:
        collected = _collect_sse(client, base, body)
        if collected is None:
            return False
        print(f"  Response preview: {collected['text'][:200]}...")
        print(f"  Chunks received: {collected['chunk_count']}")
        print(f"  Model: {collected['model']}")
        return collected["has_done"] and len(collected["text"]) > 50
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_tool_call_history(client: httpx.Client, base: str) -> bool:
    """Test with tool call history in messages — the pattern that caused Gemini 400s."""
    print("\n=== Test: Tool Call History (Gemini thought_signature regression) ===")
    body = {
        "model": "auto",
        "stream": True,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant with tool access."},
            {"role": "user", "content": "What's the weather in San Francisco?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "San Francisco, CA"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc123",
                "content": '{"temperature": 62, "condition": "Partly Cloudy"}',
            },
            {
                "role": "assistant",
                "content": "The weather in San Francisco is 62°F and partly cloudy.",
            },
            {"role": "user", "content": "Thanks! What about New York?"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City and state"},
                        },
                        "required": ["location"],
                    },
                },
            }
        ],
    }
    try:
        collected = _collect_sse(client, base, body)
        if collected is None:
            return False
        print(f"  Response: {collected['text'][:200]}")
        print(f"  Tool calls in response: {collected['tool_calls']}")
        print(f"  Model: {collected['model']}")
        # Should either give a text response or make a tool call — not error
        return collected["has_done"]
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_tool_result_no_name(client: httpx.Client, base: str) -> bool:
    """Test tool result without 'name' field — the pattern that caused empty name error."""
    print("\n=== Test: Tool Result Without Name Field ===")
    body = {
        "model": "auto",
        "stream": True,
        "messages": [
            {"role": "user", "content": "List files in the current directory"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_exec_001",
                        "type": "function",
                        "function": {
                            "name": "exec",
                            "arguments": '{"command": "ls -la"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_exec_001",
                # No "name" field — this is what OpenClaw actually sends
                "content": "total 42\ndrwxr-xr-x  5 user staff  160 Mar 15 10:00 .\ndrwxr-xr-x  3 user staff   96 Mar 14 09:00 ..\n-rw-r--r--  1 user staff 1234 Mar 15 10:00 file.txt",
            },
            {"role": "user", "content": "What files did you find?"},
        ],
    }
    try:
        collected = _collect_sse(client, base, body)
        if collected is None:
            return False
        print(f"  Response: {collected['text'][:200]}")
        print(f"  Model: {collected['model']}")
        return collected["has_done"] and len(collected["text"]) > 0
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_non_streaming(client: httpx.Client, base: str) -> bool:
    """Test non-streaming request (stream: false or absent)."""
    print("\n=== Test: Non-Streaming Request ===")
    body = {
        "model": "auto",
        "stream": False,
        "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    }
    try:
        r = client.post(
            f"{base}/v1/chat/completions",
            json=body,
            timeout=120,
        )
        print(f"  Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            model = data.get("model", "?")
            print(f"  Model: {model}")
            print(f"  Response: {text[:200]}")
            return True
        else:
            print(f"  Error: {r.text[:300]}")
            return False
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def _collect_sse(
    client: httpx.Client, base: str, body: dict
) -> dict | None:
    """Send a streaming request and collect SSE chunks into a result dict."""
    text_parts = []
    tool_calls = []
    chunk_count = 0
    has_done = False
    model = "?"

    with client.stream(
        "POST",
        f"{base}/v1/chat/completions",
        json=body,
        timeout=120,
    ) as resp:
        if resp.status_code != 200:
            body_text = ""
            for chunk in resp.iter_text():
                body_text += chunk
            print(f"  HTTP {resp.status_code}: {body_text[:300]}")
            return None

        content_type = resp.headers.get("content-type", "")
        print(f"  Content-Type: {content_type}")

        for line in resp.iter_lines():
            line = line.strip()
            if not line:
                continue
            if line == "data: [DONE]":
                has_done = True
                continue
            if not line.startswith("data: "):
                continue

            json_str = line[len("data: "):]
            try:
                chunk = json.loads(json_str)
            except json.JSONDecodeError:
                print(f"  Bad JSON chunk: {json_str[:100]}")
                continue

            chunk_count += 1
            model = chunk.get("model", model)

            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("content"):
                    text_parts.append(delta["content"])
                if delta.get("tool_calls"):
                    tool_calls.extend(delta["tool_calls"])

    return {
        "text": "".join(text_parts),
        "tool_calls": tool_calls,
        "chunk_count": chunk_count,
        "has_done": has_done,
        "model": model,
    }


def main():
    parser = argparse.ArgumentParser(description="Test semantic-claw-router locally")
    parser.add_argument("--base-url", default=DEFAULT_BASE, help="Router base URL")
    parser.add_argument(
        "--test",
        choices=["health", "models", "simple", "complex", "tools", "no-name", "non-stream", "all"],
        default="all",
        help="Which test to run",
    )
    args = parser.parse_args()

    client = httpx.Client()
    base = args.base_url.rstrip("/")

    tests = {
        "health": ("Health", lambda: test_health(client, base)),
        "models": ("Models", lambda: test_models(client, base)),
        "simple": ("Simple Streaming", lambda: test_simple_streaming(client, base)),
        "complex": ("Complex Streaming", lambda: test_complex_streaming(client, base)),
        "tools": ("Tool Call History", lambda: test_tool_call_history(client, base)),
        "no-name": ("Tool Result No Name", lambda: test_tool_result_no_name(client, base)),
        "non-stream": ("Non-Streaming", lambda: test_non_streaming(client, base)),
    }

    if args.test == "all":
        run_tests = list(tests.keys())
    else:
        run_tests = [args.test]

    results = {}
    for key in run_tests:
        name, fn = tests[key]
        start = time.monotonic()
        try:
            passed = fn()
        except KeyboardInterrupt:
            print("\n  Interrupted")
            passed = False
        elapsed = (time.monotonic() - start) * 1000
        status = "PASS" if passed else "FAIL"
        results[name] = (status, elapsed)
        print(f"  => {status} ({elapsed:.0f}ms)")

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    all_pass = True
    for name, (status, elapsed) in results.items():
        icon = "+" if status == "PASS" else "-"
        print(f"  [{icon}] {name}: {status} ({elapsed:.0f}ms)")
        if status != "PASS":
            all_pass = False

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
