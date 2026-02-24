#!/usr/bin/env python3
"""Comparative classifier test — ClawRouter vs vLLM Semantic Router vs Combined.

Runs the same diverse set of prompts through each classifier independently:

  1. **ClawRouter approach** — Fast-path regex/heuristic classifier (15 weighted
     dimensions, <1ms, no ML dependencies). Confidence threshold set to 0.0 so
     it always returns a result instead of punting to fallback.

  2. **vLLM Semantic Router approach** — Sentence-embedding classifier using
     all-MiniLM-L6-v2. Embeds the request and compares against pre-defined
     anchor prompts via cosine similarity (~10-20ms).

  3. **Combined (Semantic Claw Router)** — Fast-path first; if confidence < 0.7,
     falls back to the semantic classifier. This is the production pipeline.

Outputs a rich comparison table showing tier, confidence, and agreement for
each prompt, plus aggregate accuracy and timing statistics.

Usage:
    # Requires sentence-transformers for the semantic classifier:
    pip install semantic-claw-router[semantic]

    # Run the comparative test:
    python examples/comparative_test.py

    # JSON output (for CI/pipelines):
    python examples/comparative_test.py --json
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Add src to path for development use
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from semantic_claw_router.config import FastPathConfig, SemanticClassifierConfig
from semantic_claw_router.router.fastpath import classify_fast_path
from semantic_claw_router.router.types import (
    ChatMessage,
    ClassificationResult,
    ComplexityTier,
    ParsedRequest,
)

# Try importing the semantic classifier — gracefully degrade if unavailable.
# We check for `sentence_transformers` directly, not just our wrapper module,
# because the wrapper always imports but returns None without the real model.
try:
    import sentence_transformers  # noqa: F401

    from semantic_claw_router.router.semantic import (
        _SemanticClassifierSingleton,
        classify_semantic,
    )

    HAS_SEMANTIC = True
except ImportError:
    HAS_SEMANTIC = False


# ── ANSI Colors ───────────────────────────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
WHITE = "\033[97m"

TIER_COLORS = {
    "SIMPLE": GREEN,
    "MEDIUM": YELLOW,
    "COMPLEX": RED,
    "REASONING": MAGENTA,
}


def _c(tier: str) -> str:
    return TIER_COLORS.get(tier, WHITE)


# ── Test Prompts ──────────────────────────────────────────────────────
#
# Each prompt has a name, expected tier, and the user message content.
# The set includes:
#   - Clear-cut cases where both classifiers should agree
#   - Boundary cases where approaches may diverge
#   - Misleading prompts (simple intent with complex vocabulary, or vice versa)

@dataclass
class TestPrompt:
    name: str
    expected_tier: str
    user_content: str
    system_content: str | None = None
    tools: list | None = None
    category: str = "standard"  # standard, boundary, misleading


TEST_PROMPTS = [
    # ── CLEAR-CUT SIMPLE ──────────────────────────────────────────
    TestPrompt(
        name="Definition lookup",
        expected_tier="SIMPLE",
        user_content="What is a Python decorator?",
        category="standard",
    ),
    TestPrompt(
        name="Simple conversion",
        expected_tier="SIMPLE",
        user_content="Convert 72°F to Celsius.",
        category="standard",
    ),
    TestPrompt(
        name="Greeting",
        expected_tier="SIMPLE",
        user_content="Hello! Can you help me today?",
        category="standard",
    ),
    TestPrompt(
        name="Factual lookup",
        expected_tier="SIMPLE",
        user_content="What is the capital of France?",
        category="standard",
    ),

    # ── CLEAR-CUT MEDIUM ──────────────────────────────────────────
    TestPrompt(
        name="Function implementation",
        expected_tier="MEDIUM",
        user_content=(
            "Write a Python function that takes a list of dictionaries and "
            "returns them sorted by a specified key, handling missing keys gracefully."
        ),
        system_content="You are an expert Python developer.",
        category="standard",
    ),
    TestPrompt(
        name="Bug fix with code",
        expected_tier="MEDIUM",
        user_content=(
            "This function returns 0 for all inputs:\n\n"
            "def factorial(n):\n"
            "    result = 0\n"
            "    for i in range(1, n+1):\n"
            "        result *= i\n"
            "    return result\n\n"
            "What's wrong and how do I fix it?"
        ),
        category="standard",
    ),
    TestPrompt(
        name="SQL query writing",
        expected_tier="MEDIUM",
        user_content=(
            "Write a SQL query that finds customers who made more than 5 orders "
            "in the last 30 days, grouped by region, with the total amount spent."
        ),
        category="standard",
    ),

    # ── CLEAR-CUT COMPLEX ─────────────────────────────────────────
    TestPrompt(
        name="Microservices architecture",
        expected_tier="COMPLEX",
        user_content=(
            "Design a microservices architecture for a real-time bidding platform. "
            "The system must handle 100K requests/second with sub-10ms latency. "
            "Define the service boundaries, data flow, message broker strategy, "
            "and circuit breaker patterns. Include a deployment strategy for "
            "Kubernetes with horizontal pod autoscaling."
        ),
        system_content="You are a senior distributed systems architect.",
        category="standard",
    ),
    TestPrompt(
        name="Multi-concern system design",
        expected_tier="COMPLEX",
        user_content=(
            "Design a distributed event-driven architecture for processing "
            "financial transactions with exactly-once delivery guarantees, "
            "horizontal scaling, and multi-region failover."
        ),
        category="standard",
    ),

    # ── CLEAR-CUT REASONING ───────────────────────────────────────
    TestPrompt(
        name="Mathematical proof",
        expected_tier="REASONING",
        user_content=(
            "Prove by contradiction that there are infinitely many prime numbers. "
            "Show each logical step of the derivation."
        ),
        category="standard",
    ),
    TestPrompt(
        name="Algorithm correctness proof",
        expected_tier="REASONING",
        user_content=(
            "Prove that Dijkstra's algorithm correctly finds the shortest path "
            "in a graph with non-negative edge weights. Use mathematical induction "
            "on the number of vertices processed. Derive the loop invariant."
        ),
        category="standard",
    ),
    TestPrompt(
        name="Complexity lower bound",
        expected_tier="REASONING",
        user_content=(
            "Prove by contradiction that comparison-based sorting algorithms "
            "have a lower bound of Ω(n log n). Use the decision tree model."
        ),
        category="standard",
    ),

    # ── BOUNDARY CASES (where classifiers may diverge) ────────────
    TestPrompt(
        name="Short but complex intent",
        expected_tier="COMPLEX",
        user_content="How do I implement distributed consensus?",
        category="boundary",
    ),
    TestPrompt(
        name="Long but simple intent",
        expected_tier="SIMPLE",
        user_content=(
            "I've been working on this project for a while now and I was just "
            "wondering, since I keep forgetting every time I look it up, "
            "what exactly is the difference between a list and a tuple in Python? "
            "I know it's probably really basic but I always mix them up."
        ),
        category="boundary",
    ),
    TestPrompt(
        name="Code keywords but simple question",
        expected_tier="SIMPLE",
        user_content="What does the 'async' keyword mean in Python?",
        category="boundary",
    ),
    TestPrompt(
        name="No code keywords but medium task",
        expected_tier="MEDIUM",
        user_content=(
            "Create a small program that takes a list of names from the user, "
            "removes any duplicates, sorts them alphabetically, and displays "
            "the final result."
        ),
        category="boundary",
    ),
    TestPrompt(
        name="Ambiguous complexity",
        expected_tier="MEDIUM",
        user_content=(
            "Explain how binary search works and write an implementation."
        ),
        category="boundary",
    ),

    # ── MISLEADING PROMPTS (intentional traps) ────────────────────
    TestPrompt(
        name="Complex vocabulary, simple question",
        expected_tier="SIMPLE",
        user_content=(
            "In the context of distributed systems and microservices architecture, "
            "what is the definition of 'eventual consistency'?"
        ),
        category="misleading",
    ),
    TestPrompt(
        name="Simple language, reasoning task",
        expected_tier="REASONING",
        user_content=(
            "Show me why you can't have a set that contains itself. "
            "Walk through each step of the logic carefully."
        ),
        category="misleading",
    ),
    TestPrompt(
        name="Casual tone, complex task",
        expected_tier="COMPLEX",
        user_content=(
            "Hey, so I need help setting up a whole CI/CD pipeline for my app. "
            "It should build Docker images, run tests, do security scanning, "
            "deploy to staging automatically, then wait for approval before prod. "
            "Oh and it needs to work with ArgoCD and handle rollbacks."
        ),
        category="misleading",
    ),
    TestPrompt(
        name="Academic framing, simple lookup",
        expected_tier="SIMPLE",
        user_content="Define the time complexity of a hash table lookup operation.",
        category="misleading",
    ),
    TestPrompt(
        name="Imperative but trivial",
        expected_tier="SIMPLE",
        user_content="List the primitive data types in JavaScript.",
        category="misleading",
    ),

    # ── AGENTIC (tool-use detection) ──────────────────────────────
    TestPrompt(
        name="Tool-use: file operations",
        expected_tier="MEDIUM",
        user_content="Read the file src/main.py and fix any bugs you find.",
        tools=[
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
        ],
        category="standard",
    ),
]


# ── Helpers ───────────────────────────────────────────────────────────


def _make_parsed_request(prompt: TestPrompt) -> ParsedRequest:
    """Convert a TestPrompt into a ParsedRequest for classification."""
    messages = []
    if prompt.system_content:
        messages.append(ChatMessage(role="system", content=prompt.system_content))
    messages.append(ChatMessage(role="user", content=prompt.user_content))
    return ParsedRequest(
        model="auto",
        messages=messages,
        tools=prompt.tools,
    )


@dataclass
class ClassifierResult:
    """Result from one classifier for one prompt."""
    tier: str
    confidence: float
    latency_ms: float
    dominant_dimension: str = ""
    available: bool = True


def _run_clawrouter(
    parsed: ParsedRequest,
    config: FastPathConfig,
) -> ClassifierResult:
    """Run ClawRouter-style fast-path classifier only."""
    start = time.perf_counter()
    result = classify_fast_path(parsed, config)
    elapsed = (time.perf_counter() - start) * 1000

    if result is None:
        # Should not happen with threshold=0.0, but just in case
        return ClassifierResult(
            tier="UNKNOWN", confidence=0.0, latency_ms=elapsed,
            dominant_dimension="none",
        )

    return ClassifierResult(
        tier=result.tier.value,
        confidence=result.confidence,
        latency_ms=elapsed,
        dominant_dimension=result.dominant_dimension,
    )


def _run_semantic_router(
    parsed: ParsedRequest,
    config: SemanticClassifierConfig,
) -> ClassifierResult:
    """Run vLLM Semantic Router-style embedding classifier only."""
    if not HAS_SEMANTIC:
        return ClassifierResult(
            tier="N/A", confidence=0.0, latency_ms=0.0,
            dominant_dimension="unavailable", available=False,
        )

    start = time.perf_counter()
    result = classify_semantic(parsed, config)
    elapsed = (time.perf_counter() - start) * 1000

    if result is None:
        return ClassifierResult(
            tier="FAILED", confidence=0.0, latency_ms=elapsed,
            dominant_dimension="error",
        )

    return ClassifierResult(
        tier=result.tier.value,
        confidence=result.confidence,
        latency_ms=elapsed,
        dominant_dimension=result.dominant_dimension,
    )


def _run_combined(
    parsed: ParsedRequest,
    fp_config: FastPathConfig,
    sem_config: SemanticClassifierConfig,
) -> ClassifierResult:
    """Run combined pipeline: fast-path first, semantic fallback if ambiguous."""
    start = time.perf_counter()

    # Stage 1: Fast-path (with real threshold)
    result = classify_fast_path(parsed, fp_config)

    source = "fast_path"
    if result is None:
        # Stage 2: Semantic fallback
        source = "semantic_fallback"
        if HAS_SEMANTIC:
            result_sem = classify_semantic(parsed, sem_config)
            if result_sem is not None:
                elapsed = (time.perf_counter() - start) * 1000
                return ClassifierResult(
                    tier=result_sem.tier.value,
                    confidence=result_sem.confidence,
                    latency_ms=elapsed,
                    dominant_dimension=f"{source}:{result_sem.dominant_dimension}",
                )

        # Stage 3: Heuristic fallback (fast-path with no threshold)
        source = "heuristic_fallback"
        fallback_config = FastPathConfig(
            enabled=True,
            confidence_threshold=0.0,
            weights=fp_config.weights,
            tier_boundaries=fp_config.tier_boundaries,
        )
        result = classify_fast_path(parsed, fallback_config)

    elapsed = (time.perf_counter() - start) * 1000

    if result is None:
        return ClassifierResult(
            tier="UNKNOWN", confidence=0.0, latency_ms=elapsed,
            dominant_dimension=source,
        )

    return ClassifierResult(
        tier=result.tier.value,
        confidence=result.confidence,
        latency_ms=elapsed,
        dominant_dimension=f"{source}:{result.dominant_dimension}",
    )


# ── Main Runner ───────────────────────────────────────────────────────


def run_comparison(*, json_output: bool = False):
    """Run all test prompts through each classifier and display results."""

    # ── Configs ───────────────────────────────────────────────────
    # ClawRouter: force always-return by setting threshold to 0.0
    clawrouter_config = FastPathConfig(
        enabled=True,
        confidence_threshold=0.0,
    )

    # Combined: uses real threshold (0.7) for fast-path
    combined_fp_config = FastPathConfig(
        enabled=True,
        confidence_threshold=0.7,
    )

    semantic_config = SemanticClassifierConfig(enabled=True)

    # ── Warm up semantic model if available ────────────────────────
    if HAS_SEMANTIC:
        warmup_req = _make_parsed_request(TestPrompt(
            name="warmup", expected_tier="SIMPLE",
            user_content="warmup query",
        ))
        classify_semantic(warmup_req, semantic_config)
        # Reset singleton timing after warmup
        print(f"  {DIM}Semantic model loaded (all-MiniLM-L6-v2){RESET}\n")

    # ── Run tests ─────────────────────────────────────────────────
    results = []

    for prompt in TEST_PROMPTS:
        parsed = _make_parsed_request(prompt)

        claw = _run_clawrouter(parsed, clawrouter_config)
        sem = _run_semantic_router(parsed, semantic_config)
        combined = _run_combined(parsed, combined_fp_config, semantic_config)

        results.append({
            "name": prompt.name,
            "expected": prompt.expected_tier,
            "category": prompt.category,
            "user_content": prompt.user_content[:80],
            "clawrouter": claw,
            "semantic": sem,
            "combined": combined,
        })

    # ── Output ────────────────────────────────────────────────────
    if json_output:
        _print_json(results)
    else:
        _print_table(results)

    return results


def _print_table(results: list[dict]):
    """Print a rich comparison table to stdout."""

    print(f"\n{BOLD}{'═'*100}")
    print("  COMPARATIVE CLASSIFIER TEST")
    print("  ClawRouter (fast-path regex) vs vLLM Semantic Router (embeddings) vs Combined")
    print(f"{'═'*100}{RESET}\n")

    # ── Per-prompt results ────────────────────────────────────────

    categories = ["standard", "boundary", "misleading"]
    category_labels = {
        "standard": "CLEAR-CUT CASES",
        "boundary": "BOUNDARY CASES (where classifiers may diverge)",
        "misleading": "MISLEADING PROMPTS (intentional traps)",
    }

    claw_correct = 0
    sem_correct = 0
    combined_correct = 0
    agreements = 0
    total = len(results)

    claw_latencies = []
    sem_latencies = []
    combined_latencies = []

    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        if not cat_results:
            continue

        print(f"  {BOLD}{CYAN}── {category_labels[cat]} {'─' * (70 - len(category_labels[cat]))}{RESET}\n")

        # Header
        print(f"  {'Prompt':<35s} {'Expected':>8s}  │  "
              f"{'ClawRouter':>10s} {'Conf':>5s}  │  "
              f"{'Semantic':>10s} {'Conf':>5s}  │  "
              f"{'Combined':>10s} {'Conf':>5s}  │ {'Agree?':>6s}")
        print(f"  {'─'*35} {'─'*8}──┼──{'─'*10}─{'─'*5}──┼──"
              f"{'─'*10}─{'─'*5}──┼──{'─'*10}─{'─'*5}──┼─{'─'*6}")

        for r in cat_results:
            name = r["name"][:35]
            expected = r["expected"]
            claw: ClassifierResult = r["clawrouter"]
            sem: ClassifierResult = r["semantic"]
            comb: ClassifierResult = r["combined"]

            # Track accuracy
            claw_match = claw.tier == expected
            sem_match = sem.tier == expected if sem.available else False
            comb_match = comb.tier == expected

            if claw_match:
                claw_correct += 1
            if sem_match:
                sem_correct += 1
            if comb_match:
                combined_correct += 1

            agree = claw.tier == sem.tier if sem.available else True
            if agree:
                agreements += 1

            claw_latencies.append(claw.latency_ms)
            if sem.available:
                sem_latencies.append(sem.latency_ms)
            combined_latencies.append(comb.latency_ms)

            # Format with colors
            claw_icon = "✓" if claw_match else "✗"
            sem_icon = "✓" if sem_match else ("─" if not sem.available else "✗")
            comb_icon = "✓" if comb_match else "✗"
            agree_icon = f"{GREEN}Yes{RESET}" if agree else f"{RED}NO{RESET} "

            claw_str = f"{_c(claw.tier)}{claw.tier:>10s}{RESET}"
            sem_str = (f"{_c(sem.tier)}{sem.tier:>10s}{RESET}"
                       if sem.available else f"{'N/A':>10s}")
            comb_str = f"{_c(comb.tier)}{comb.tier:>10s}{RESET}"

            print(
                f"  {name:<35s} {_c(expected)}{expected:>8s}{RESET}  │  "
                f"{claw_str} {claw.confidence:>4.2f}{claw_icon} │  "
                f"{sem_str} {sem.confidence:>4.2f}{sem_icon} │  "
                f"{comb_str} {comb.confidence:>4.2f}{comb_icon} │ {agree_icon}"
            )

        print()

    # ── Summary Statistics ────────────────────────────────────────

    sem_total = sum(1 for r in results if r["semantic"].available)

    print(f"\n{BOLD}{'═'*100}")
    print(f"  SUMMARY")
    print(f"{'═'*100}{RESET}\n")

    print(f"  {BOLD}Accuracy (correct tier classification):{RESET}")
    claw_pct = claw_correct / total * 100
    comb_pct = combined_correct / total * 100

    claw_bar = "█" * int(claw_pct / 5)
    comb_bar = "█" * int(comb_pct / 5)

    print(f"    ClawRouter (regex):    {claw_bar:<20s} {claw_correct}/{total} ({claw_pct:.0f}%)")
    if sem_total > 0:
        sem_pct = sem_correct / sem_total * 100
        sem_bar = "█" * int(sem_pct / 5)
        print(f"    Semantic (embedding):  {sem_bar:<20s} {sem_correct}/{sem_total} ({sem_pct:.0f}%)")
    else:
        print(f"    Semantic (embedding):  {'─' * 20} N/A (not installed)")
    print(f"    Combined (pipeline):   {comb_bar:<20s} {combined_correct}/{total} ({comb_pct:.0f}%)")

    print(f"\n  {BOLD}Agreement (ClawRouter ↔ Semantic):{RESET}")
    agree_pct = agreements / total * 100
    print(f"    Agree on tier:         {agreements}/{total} ({agree_pct:.0f}%)")

    print(f"\n  {BOLD}Latency (median):{RESET}")
    claw_latencies.sort()
    combined_latencies.sort()
    claw_p50 = claw_latencies[len(claw_latencies) // 2] if claw_latencies else 0
    comb_p50 = combined_latencies[len(combined_latencies) // 2] if combined_latencies else 0

    print(f"    ClawRouter:            {claw_p50:.3f} ms")
    if sem_latencies:
        sem_latencies.sort()
        sem_p50 = sem_latencies[len(sem_latencies) // 2]
        print(f"    Semantic:              {sem_p50:.3f} ms")
    else:
        print(f"    Semantic:              N/A (not installed)")
    print(f"    Combined:              {comb_p50:.3f} ms")

    # ── Disagreement Analysis ─────────────────────────────────────

    disagreements = [
        r for r in results
        if r["semantic"].available and r["clawrouter"].tier != r["semantic"].tier
    ]
    if disagreements:
        print(f"\n  {BOLD}Disagreements (ClawRouter ≠ Semantic):{RESET}")
        for r in disagreements:
            claw = r["clawrouter"]
            sem = r["semantic"]
            expected = r["expected"]
            claw_right = "✓" if claw.tier == expected else "✗"
            sem_right = "✓" if sem.tier == expected else "✗"
            print(
                f"    • {r['name']:<35s} Expected: {_c(expected)}{expected}{RESET}  "
                f"ClawRouter: {_c(claw.tier)}{claw.tier}{RESET}({claw.confidence:.2f}){claw_right}  "
                f"Semantic: {_c(sem.tier)}{sem.tier}{RESET}({sem.confidence:.2f}){sem_right}"
            )

    # ── Per-Category Breakdown ────────────────────────────────────

    print(f"\n  {BOLD}Per-Category Accuracy:{RESET}")
    for cat in ["standard", "boundary", "misleading"]:
        cat_results = [r for r in results if r["category"] == cat]
        if not cat_results:
            continue
        cat_n = len(cat_results)
        c_claw = sum(1 for r in cat_results if r["clawrouter"].tier == r["expected"])
        c_sem = sum(1 for r in cat_results if r["semantic"].available and r["semantic"].tier == r["expected"])
        c_sem_n = sum(1 for r in cat_results if r["semantic"].available)
        c_comb = sum(1 for r in cat_results if r["combined"].tier == r["expected"])

        print(f"    {cat:>12s}:  ClawRouter {c_claw}/{cat_n}  "
              f"Semantic {c_sem}/{c_sem_n}  Combined {c_comb}/{cat_n}")

    print(f"\n{'═'*100}\n")


def _print_json(results: list[dict]):
    """Print results as JSON for CI/pipeline consumption."""
    output = {
        "test_name": "comparative_classifier_test",
        "semantic_available": HAS_SEMANTIC,
        "prompts": [],
        "summary": {},
    }

    total = len(results)
    claw_correct = 0
    sem_correct = 0
    combined_correct = 0

    for r in results:
        claw: ClassifierResult = r["clawrouter"]
        sem: ClassifierResult = r["semantic"]
        comb: ClassifierResult = r["combined"]

        if claw.tier == r["expected"]:
            claw_correct += 1
        if sem.available and sem.tier == r["expected"]:
            sem_correct += 1
        if comb.tier == r["expected"]:
            combined_correct += 1

        output["prompts"].append({
            "name": r["name"],
            "category": r["category"],
            "expected_tier": r["expected"],
            "clawrouter": {
                "tier": claw.tier,
                "confidence": round(claw.confidence, 4),
                "latency_ms": round(claw.latency_ms, 3),
                "correct": claw.tier == r["expected"],
            },
            "semantic": {
                "tier": sem.tier if sem.available else None,
                "confidence": round(sem.confidence, 4) if sem.available else None,
                "latency_ms": round(sem.latency_ms, 3) if sem.available else None,
                "correct": sem.tier == r["expected"] if sem.available else None,
                "available": sem.available,
            },
            "combined": {
                "tier": comb.tier,
                "confidence": round(comb.confidence, 4),
                "latency_ms": round(comb.latency_ms, 3),
                "correct": comb.tier == r["expected"],
                "source": comb.dominant_dimension,
            },
            "agree": claw.tier == sem.tier if sem.available else None,
        })

    sem_total = sum(1 for r in results if r["semantic"].available)
    output["summary"] = {
        "total_prompts": total,
        "clawrouter_accuracy": round(claw_correct / total, 4),
        "semantic_accuracy": round(sem_correct / sem_total, 4) if sem_total > 0 else None,
        "combined_accuracy": round(combined_correct / total, 4),
        "clawrouter_correct": claw_correct,
        "semantic_correct": sem_correct,
        "combined_correct": combined_correct,
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    json_mode = "--json" in sys.argv
    run_comparison(json_output=json_mode)
