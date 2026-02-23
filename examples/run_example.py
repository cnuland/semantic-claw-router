#!/usr/bin/env python3
"""Example runner for the Semantic Claw Router.

Starts the router server and sends a comprehensive set of simulated
OpenClaw-style requests to demonstrate tier-based routing between
local Qwen3 (SIMPLE/MEDIUM) and Gemini (COMPLEX/REASONING).

Usage:
    python examples/run_example.py

The script:
1. Starts the router on localhost:8080
2. Sends 20+ simulated requests across all complexity tiers
3. Shows routing decisions, model selection, and latency for each
4. Prints aggregate metrics at the end
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from semantic_claw_router.config import RouterConfig
from semantic_claw_router.server import SemanticRouter

# ── OpenClaw-Style Simulated Prompts ─────────────────────────────────
#
# These simulate the types of requests an AI coding assistant like
# OpenClaw would send. They span the full complexity spectrum:
#
# - SIMPLE: Quick lookups, definitions, simple transformations
# - MEDIUM: Code generation, moderate debugging, documentation
# - COMPLEX: Multi-file refactoring, architecture, system design
# - REASONING: Proofs, formal analysis, complex debugging chains
# - AGENTIC: Tool-use workflows, multi-step autonomous tasks

OPENCLAW_PROMPTS = [
    # ── SIMPLE tier (should route to Qwen3) ──────────────────────
    {
        "name": "Definition lookup",
        "expected_tier": "SIMPLE",
        "body": {
            "model": "auto",
            "messages": [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": "What is a Python decorator?"},
            ],
            "max_tokens": 150,
        },
    },
    {
        "name": "Simple conversion",
        "expected_tier": "SIMPLE",
        "body": {
            "model": "auto",
            "messages": [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": "Convert this hex color #FF5733 to RGB values."},
            ],
            "max_tokens": 50,
        },
    },
    {
        "name": "Quick syntax question",
        "expected_tier": "SIMPLE",
        "body": {
            "model": "auto",
            "messages": [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": "What is the difference between == and === in JavaScript?"},
            ],
            "max_tokens": 150,
        },
    },
    {
        "name": "Hello world",
        "expected_tier": "SIMPLE",
        "body": {
            "model": "auto",
            "messages": [
                {"role": "user", "content": "Hello! Can you help me today?"},
            ],
            "max_tokens": 50,
        },
    },

    # ── MEDIUM tier (should route to Qwen3) ──────────────────────
    {
        "name": "Function implementation",
        "expected_tier": "MEDIUM",
        "body": {
            "model": "auto",
            "messages": [
                {"role": "system", "content": "You are an expert Python developer."},
                {"role": "user", "content": (
                    "Write a Python function called `merge_sorted_lists` that takes "
                    "two sorted lists of integers and returns a single sorted list. "
                    "Use the merge step from merge sort."
                )},
            ],
            "max_tokens": 300,
        },
    },
    {
        "name": "Bug fix request",
        "expected_tier": "MEDIUM",
        "body": {
            "model": "auto",
            "messages": [
                {"role": "system", "content": "You are an expert debugger."},
                {"role": "user", "content": (
                    "This function is supposed to return the factorial of n but it "
                    "returns 0 for all inputs:\n\n"
                    "def factorial(n):\n"
                    "    result = 0\n"
                    "    for i in range(1, n+1):\n"
                    "        result *= i\n"
                    "    return result\n\n"
                    "What's wrong and how do I fix it?"
                )},
            ],
            "max_tokens": 200,
        },
    },
    {
        "name": "Code review",
        "expected_tier": "MEDIUM",
        "body": {
            "model": "auto",
            "messages": [
                {"role": "system", "content": "You are a senior code reviewer."},
                {"role": "user", "content": (
                    "Review this Python class for best practices:\n\n"
                    "class UserManager:\n"
                    "    def __init__(self):\n"
                    "        self.users = []\n\n"
                    "    def add_user(self, name, email):\n"
                    "        self.users.append({'name': name, 'email': email})\n\n"
                    "    def get_user(self, name):\n"
                    "        for u in self.users:\n"
                    "            if u['name'] == name:\n"
                    "                return u\n"
                    "        return None\n"
                )},
            ],
            "max_tokens": 300,
        },
    },
    {
        "name": "Unit test generation",
        "expected_tier": "MEDIUM",
        "body": {
            "model": "auto",
            "messages": [
                {"role": "system", "content": "You are a testing expert."},
                {"role": "user", "content": (
                    "Write pytest unit tests for this function:\n\n"
                    "def is_palindrome(s: str) -> bool:\n"
                    "    s = s.lower().replace(' ', '')\n"
                    "    return s == s[::-1]\n"
                )},
            ],
            "max_tokens": 300,
        },
    },
    {
        "name": "Documentation generation",
        "expected_tier": "MEDIUM",
        "body": {
            "model": "auto",
            "messages": [
                {"role": "system", "content": "You are a technical writer."},
                {"role": "user", "content": (
                    "Write a docstring for this function:\n\n"
                    "async def fetch_and_cache(url, cache, ttl=300):\n"
                    "    cached = cache.get(url)\n"
                    "    if cached and time.time() - cached['ts'] < ttl:\n"
                    "        return cached['data']\n"
                    "    resp = await httpx.get(url)\n"
                    "    data = resp.json()\n"
                    "    cache.set(url, {'data': data, 'ts': time.time()})\n"
                    "    return data\n"
                )},
            ],
            "max_tokens": 200,
        },
    },

    # ── COMPLEX tier (should route to Gemini) ────────────────────
    {
        "name": "API design with constraints",
        "expected_tier": "COMPLEX",
        "body": {
            "model": "auto",
            "messages": [
                {"role": "system", "content": "You are a senior backend architect."},
                {"role": "user", "content": (
                    "Design a RESTful API for a task management system. "
                    "First, define the database schema with PostgreSQL. "
                    "Then, implement the CRUD endpoints using FastAPI. "
                    "Ensure proper authentication with JWT tokens. "
                    "The system must handle concurrent updates with optimistic locking. "
                    "Return the response as JSON with a clear schema."
                )},
            ],
            "max_tokens": 800,
        },
    },
    {
        "name": "Distributed systems debugging",
        "expected_tier": "COMPLEX",
        "body": {
            "model": "auto",
            "messages": [
                {"role": "system", "content": "You are a distributed systems expert."},
                {"role": "user", "content": (
                    "Our Kubernetes microservice architecture is experiencing "
                    "intermittent 503 errors. The service mesh (Istio) shows "
                    "connection pool exhaustion on service-A → service-B calls. "
                    "Analyze the possible causes considering: circuit breaker "
                    "configuration, connection keep-alive settings, pod autoscaling "
                    "behavior, and DNS resolution caching. Design a step-by-step "
                    "debugging approach and implement the fix."
                )},
            ],
            "max_tokens": 800,
        },
    },
    {
        "name": "Multi-file refactoring",
        "expected_tier": "COMPLEX",
        "body": {
            "model": "auto",
            "messages": [
                {"role": "system", "content": "You are a refactoring specialist."},
                {"role": "user", "content": (
                    "Refactor this monolithic Django view into a clean architecture. "
                    "First, extract the business logic into a service layer. "
                    "Then, create proper serializers for input validation. "
                    "Next, implement the repository pattern for database access. "
                    "Ensure backward compatibility with existing API consumers. "
                    "The code must pass all existing integration tests."
                )},
            ],
            "max_tokens": 800,
        },
    },

    # ── REASONING tier (should route to Gemini) ──────────────────
    {
        "name": "Algorithm proof",
        "expected_tier": "REASONING",
        "body": {
            "model": "auto",
            "messages": [
                {"role": "system", "content": "You are a computer science professor."},
                {"role": "user", "content": (
                    "Prove that Dijkstra's algorithm correctly finds the shortest "
                    "path in a graph with non-negative edge weights. Use mathematical "
                    "induction on the number of vertices processed. Derive the "
                    "loop invariant and prove it holds at each step."
                )},
            ],
            "max_tokens": 1000,
        },
    },
    {
        "name": "Complexity analysis",
        "expected_tier": "REASONING",
        "body": {
            "model": "auto",
            "messages": [
                {"role": "system", "content": "You are an algorithms expert."},
                {"role": "user", "content": (
                    "Prove by contradiction that comparison-based sorting algorithms "
                    "have a lower bound of Ω(n log n). Use the decision tree model "
                    "and derive the minimum height of the tree."
                )},
            ],
            "max_tokens": 800,
        },
    },
    {
        "name": "Formal verification",
        "expected_tier": "REASONING",
        "body": {
            "model": "auto",
            "messages": [
                {"role": "system", "content": "You are a formal methods expert."},
                {"role": "user", "content": (
                    "Given this concurrent queue implementation, prove that it is "
                    "linearizable. Consider all possible interleavings of enqueue "
                    "and dequeue operations. Derive the linearization points and "
                    "prove the correctness of the CAS-based synchronization."
                )},
            ],
            "max_tokens": 800,
        },
    },

    # ── AGENTIC requests (tool-use, should detect agentic) ───────
    {
        "name": "Agentic: file read + fix",
        "expected_tier": "MEDIUM",
        "body": {
            "model": "auto",
            "messages": [
                {"role": "system", "content": (
                    "You are an AI coding assistant with access to file system tools. "
                    "Use the provided tools to read files, make edits, and run commands."
                )},
                {"role": "user", "content": "Read the file src/main.py and fix any bugs you find."},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read the contents of a file",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "description": "Write content to a file",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "content": {"type": "string"},
                            },
                        },
                    },
                },
            ],
            "max_tokens": 500,
        },
    },
    {
        "name": "Agentic: test and debug loop",
        "expected_tier": "MEDIUM",
        "body": {
            "model": "auto",
            "messages": [
                {"role": "system", "content": (
                    "You are an AI coding assistant. Use the tools to run tests, "
                    "read code, and fix issues iteratively."
                )},
                {"role": "user", "content": (
                    "Run the tests in tests/test_auth.py. If any fail, read the "
                    "relevant source code, debug the issue, fix it, and run the "
                    "tests again. Try again until all tests pass."
                )},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "run_command",
                        "description": "Execute a shell command",
                        "parameters": {
                            "type": "object",
                            "properties": {"command": {"type": "string"}},
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a file",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "edit_file",
                        "description": "Edit a file",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "old": {"type": "string"},
                                "new": {"type": "string"},
                            },
                        },
                    },
                },
            ],
            "max_tokens": 500,
        },
    },

    # ── Session pinning test (multi-turn) ────────────────────────
    {
        "name": "Multi-turn conversation (turn 1)",
        "expected_tier": "SIMPLE",
        "headers": {"x-session-id": "example-session-1"},
        "body": {
            "model": "auto",
            "messages": [
                {"role": "user", "content": "Hi, I'm working on a Python project."},
            ],
            "max_tokens": 100,
        },
    },
    {
        "name": "Multi-turn conversation (turn 2 — pinned)",
        "expected_tier": "SIMPLE",
        "headers": {"x-session-id": "example-session-1"},
        "body": {
            "model": "auto",
            "messages": [
                {"role": "user", "content": "Hi, I'm working on a Python project."},
                {"role": "assistant", "content": "I'd be happy to help!"},
                {"role": "user", "content": "What testing framework should I use?"},
            ],
            "max_tokens": 100,
        },
    },

    # ── Dedup test (same request twice) ──────────────────────────
    {
        "name": "Dedup test (original)",
        "expected_tier": "SIMPLE",
        "body": {
            "model": "auto",
            "messages": [
                {"role": "user", "content": "What is the time complexity of binary search?"},
            ],
            "max_tokens": 50,
        },
    },
    {
        "name": "Dedup test (duplicate — should be cached)",
        "expected_tier": "SIMPLE",
        "body": {
            "model": "auto",
            "messages": [
                {"role": "user", "content": "What is the time complexity of binary search?"},
            ],
            "max_tokens": 50,
        },
    },
]


def _tier_color(tier: str) -> str:
    colors = {
        "SIMPLE": "\033[92m",    # Green
        "MEDIUM": "\033[93m",    # Yellow
        "COMPLEX": "\033[91m",   # Red
        "REASONING": "\033[95m", # Magenta
    }
    return colors.get(tier, "\033[0m")


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"


async def run_examples():
    """Run all example prompts through the router."""
    print(f"\n{BOLD}{'='*70}")
    print("  Semantic Claw Router — Example Runner")
    print(f"{'='*70}{RESET}\n")

    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    config = RouterConfig.from_yaml(str(config_path))
    config.observability.log_level = "WARNING"  # Quiet for examples

    router = SemanticRouter(config)

    print(f"  Models configured:")
    for m in config.models:
        print(f"    • {m.name} ({m.provider}) → {m.endpoint[:60]}...")
    print(f"\n  Tier mapping:")
    for tier, model in config.default_tier_models.items():
        color = _tier_color(tier)
        print(f"    {color}{tier}{RESET} → {model}")
    print(f"\n{'─'*70}\n")

    results = []
    total_start = time.monotonic()

    for i, prompt in enumerate(OPENCLAW_PROMPTS, 1):
        name = prompt["name"]
        expected = prompt["expected_tier"]
        headers = prompt.get("headers", {})
        body = prompt["body"]

        print(f"  [{i:2d}/{len(OPENCLAW_PROMPTS)}] {BOLD}{name}{RESET}")
        print(f"       Expected: {_tier_color(expected)}{expected}{RESET}")

        start = time.monotonic()
        response = await router.route_request(body, headers)
        elapsed = (time.monotonic() - start) * 1000

        actual_tier = response.headers.get("x-scr-tier", "?")
        actual_model = response.headers.get("x-scr-model", "?")
        source = response.headers.get("x-scr-source", "?")
        deduped = response.headers.get("x-scr-dedup", "false")
        is_agentic = "tools" in body

        tier_color = _tier_color(actual_tier)
        match = "✓" if actual_tier == expected or deduped == "true" else "≈"

        # Extract response preview
        preview = ""
        if response.body and "choices" in response.body:
            msg = response.body["choices"][0].get("message", {})
            content = msg.get("content") or ""
            if not content and msg.get("tool_calls"):
                content = f"[tool_call: {msg['tool_calls'][0].get('function', {}).get('name', '?')}]"
            preview = content[:80].replace("\n", " ")

        print(f"       Actual:   {tier_color}{actual_tier}{RESET} → {actual_model} ({source})")
        print(f"       Latency:  {elapsed:.0f}ms  {DIM}Dedup: {deduped}  Agentic: {is_agentic}{RESET}")
        if preview:
            print(f"       Response: {DIM}\"{preview}...\"{RESET}")
        print(f"       Result:   {match}")
        print()

        results.append({
            "name": name,
            "expected": expected,
            "actual": actual_tier,
            "model": actual_model,
            "source": source,
            "latency_ms": elapsed,
            "deduped": deduped == "true",
            "status": response.status_code,
        })

    total_elapsed = (time.monotonic() - total_start) * 1000

    # ── Print summary ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  {BOLD}RESULTS SUMMARY{RESET}")
    print(f"{'='*70}\n")

    # Metrics
    metrics = router.metrics.get_summary()
    print(f"  Total requests:  {metrics['total_requests']}")
    print(f"  Total time:      {total_elapsed:.0f}ms")
    print(f"  Avg latency:     {metrics['latency']['mean_ms']:.0f}ms")
    print(f"  P50 latency:     {metrics['latency']['p50_ms']:.0f}ms")
    print(f"  P99 latency:     {metrics['latency']['p99_ms']:.0f}ms")

    # Tier distribution
    print(f"\n  Tier distribution:")
    for tier in ["SIMPLE", "MEDIUM", "COMPLEX", "REASONING"]:
        count = metrics.get("tier_distribution", {}).get(tier, 0)
        color = _tier_color(tier)
        bar = "█" * count
        print(f"    {color}{tier:10s}{RESET} {bar} ({count})")

    # Model distribution
    print(f"\n  Model distribution:")
    for model, count in metrics.get("model_distribution", {}).items():
        bar = "█" * count
        print(f"    {model:25s} {bar} ({count})")

    # Routing source distribution
    print(f"\n  Routing source:")
    for source, count in metrics.get("routing_source_distribution", {}).items():
        print(f"    {source:25s} ({count})")

    # Dedup stats
    if router.dedup:
        print(f"\n  Dedup cache: {router.dedup.stats}")

    # Session stats
    if router.sessions:
        print(f"  Sessions:    {router.sessions.stats}")

    # Classification accuracy
    correct = sum(
        1 for r in results
        if r["actual"] == r["expected"] or r["deduped"]
    )
    total = len(results)
    print(f"\n  Classification accuracy: {correct}/{total} ({correct/total*100:.0f}%)")

    # Failed requests
    failed = [r for r in results if r["status"] != 200]
    if failed:
        print(f"\n  ⚠ Failed requests ({len(failed)}):")
        for f in failed:
            print(f"    - {f['name']}: HTTP {f['status']}")

    print(f"\n{'='*70}\n")

    await router.close()


if __name__ == "__main__":
    asyncio.run(run_examples())
