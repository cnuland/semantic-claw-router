"""Fast-path pre-classifier — 14-dimension weighted scoring.

Inspired by ClawRouter (https://github.com/BlockRunAI/ClawRouter).
Original concept by BlockRun under MIT License.
Re-implemented in Python for the semantic-claw-router pipeline.

This classifier handles 70-80% of requests in <1ms by using lightweight
heuristic scoring across 14 weighted dimensions, avoiding expensive
neural inference for obvious cases.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

from .types import ClassificationResult, ComplexityTier, ParsedRequest

if TYPE_CHECKING:
    from ..config import FastPathConfig

# ── Keyword sets for each dimension ──────────────────────────────────

_CODE_KEYWORDS = re.compile(
    r"\b(function|class|def|import|from|async|await|return|const|let|var|"
    r"interface|struct|enum|impl|pub|fn|func|package|module|"
    r"if|else|for|while|switch|case|try|catch|throw|raise|"
    r"int|float|string|bool|void|null|None|undefined|"
    r"print|console\.log|fmt\.Println|System\.out)\b",
    re.IGNORECASE,
)

_REASONING_KEYWORDS = re.compile(
    r"\b(prove|theorem|derive|derivation|proof|induction|contradiction|"
    r"chain of thought|step by step|think through|reason about|"
    r"mathematical|formal logic|axiom|lemma|corollary|"
    r"therefore|hence|thus|it follows|QED|"
    r"analyze carefully|consider all|exhaustive)\b",
    re.IGNORECASE,
)

_TECHNICAL_KEYWORDS = re.compile(
    r"\b(algorithm|optimize|optimization|kubernetes|docker|"
    r"distributed|concurrent|parallelism|microservice|"
    r"database|SQL|NoSQL|REST|GraphQL|gRPC|"
    r"architecture|scalab|latency|throughput|"
    r"encryption|authentication|authorization|"
    r"neural network|transformer|gradient|backprop|"
    r"TCP|UDP|HTTP|DNS|TLS|SSL|"
    r"polymorphism|inheritance|abstraction|encapsulation)\b",
    re.IGNORECASE,
)

_CREATIVE_KEYWORDS = re.compile(
    r"\b(story|poem|creative|imagine|brainstorm|"
    r"narrative|fiction|character|dialogue|"
    r"write me a|compose|craft|invent|"
    r"metaphor|allegory|symbolism)\b",
    re.IGNORECASE,
)

_SIMPLE_KEYWORDS = re.compile(
    r"\b(what is|define|translate|hello|hi there|"
    r"who is|when was|where is|how many|"
    r"list of|name the|tell me about|"
    r"convert|calculate|sum of|"
    r"yes or no|true or false)\b",
    re.IGNORECASE,
)

_MULTI_STEP_PATTERNS = re.compile(
    r"(first\b.{5,60}\bthen\b|step \d|"
    r"1[\.\)]\s|next\b.{5,40}\bafter|"
    r"begin by|start with.{5,40}then|"
    r"phase \d|stage \d|"
    r"first,?\s.{5,60}second,?\s|"
    r"follow these steps)",
    re.IGNORECASE,
)

_IMPERATIVE_KEYWORDS = re.compile(
    r"\b(implement|design|analyze|build|create|develop|"
    r"construct|architect|engineer|refactor|"
    r"debug|fix|resolve|diagnose|troubleshoot|"
    r"deploy|configure|set up|install|migrate)\b",
    re.IGNORECASE,
)

_CONSTRAINT_KEYWORDS = re.compile(
    r"\b(must|ensure|constraint|require|within \d|"
    r"at most|at least|no more than|"
    r"limited to|bounded by|subject to|"
    r"maximum|minimum|exactly \d|"
    r"compatible with|compliant|conform)\b",
    re.IGNORECASE,
)

_OUTPUT_FORMAT_KEYWORDS = re.compile(
    r"\b(json|yaml|yml|csv|xml|html|markdown|"
    r"table format|formatted as|output as|"
    r"return as|respond in|give me a table|"
    r"structured output|schema)\b",
    re.IGNORECASE,
)

_REFERENCE_PATTERNS = re.compile(
    r"\b(above|previous|as mentioned|referring to|"
    r"earlier|the code above|the function above|"
    r"based on the|given the|considering the|"
    r"the .{3,20} mentioned|the .{3,20} described)\b",
    re.IGNORECASE,
)

_NEGATION_PATTERNS = re.compile(
    r"\b(not|don't|doesn't|shouldn't|wouldn't|"
    r"without|except|exclude|never|"
    r"avoid|instead of|rather than|"
    r"but not|anything but)\b",
    re.IGNORECASE,
)

_DOMAIN_KEYWORDS = re.compile(
    r"\b(patient|diagnosis|clinical|prescription|"
    r"plaintiff|defendant|jurisdiction|statute|"
    r"quantum|photon|wavelength|molecular|"
    r"portfolio|derivative|hedge|equity|"
    r"topology|manifold|Hilbert|eigenvalue|"
    r"phylogenetic|genome|protein|enzyme)\b",
    re.IGNORECASE,
)

_AGENTIC_KEYWORDS = re.compile(
    r"\b(read_file|write_file|edit_file|create_file|"
    r"run_command|execute|bash|terminal|shell|"
    r"search_files|list_files|glob|grep|"
    r"try again|retry|fix the|debug the|"
    r"test and fix|run the tests|"
    r"tool_call|function_call)\b",
    re.IGNORECASE,
)


def _count_matches(pattern: re.Pattern, text: str) -> int:
    return len(pattern.findall(text))


def _score_token_count(request: ParsedRequest) -> float:
    """Score based on estimated token count. Short → negative, long → positive."""
    tokens = request.estimated_tokens
    if tokens < 50:
        return -0.8
    if tokens < 100:
        return -0.4
    if tokens < 300:
        return 0.0
    if tokens < 1000:
        return 0.3
    if tokens < 3000:
        return 0.6
    return 0.9


def _score_code_presence(request: ParsedRequest) -> float:
    """Score based on code-related keyword density."""
    text = request.all_messages_text
    if not text:
        return 0.0
    count = _count_matches(_CODE_KEYWORDS, text)
    density = count / max(len(text.split()), 1)
    return min(density * 15.0, 1.0)


def _score_reasoning_markers(request: ParsedRequest) -> float:
    """Score reasoning keywords in USER messages only (not system prompts)."""
    text = request.user_messages_text
    if not text:
        return 0.0
    count = _count_matches(_REASONING_KEYWORDS, text)
    if count >= 2:
        return 1.0  # Strong reasoning signal
    if count == 1:
        return 0.6
    return 0.0


def _score_technical_terms(request: ParsedRequest) -> float:
    text = request.all_messages_text
    if not text:
        return 0.0
    count = _count_matches(_TECHNICAL_KEYWORDS, text)
    density = count / max(len(text.split()), 1)
    return min(density * 12.0, 1.0)


def _score_creative_markers(request: ParsedRequest) -> float:
    text = request.all_messages_text
    count = _count_matches(_CREATIVE_KEYWORDS, text)
    return min(count * 0.3, 1.0)


def _score_simple_indicators(request: ParsedRequest) -> float:
    """Simple indicators produce NEGATIVE scores (lower complexity)."""
    text = request.user_messages_text
    count = _count_matches(_SIMPLE_KEYWORDS, text)
    return max(count * -0.5, -1.0)  # Clamp to [-1, 0]


def _score_multi_step(request: ParsedRequest) -> float:
    text = request.all_messages_text
    count = _count_matches(_MULTI_STEP_PATTERNS, text)
    return min(count * 0.4, 1.0)


def _score_question_complexity(request: ParsedRequest) -> float:
    text = request.user_messages_text
    q_count = text.count("?")
    if q_count == 0:
        return -0.2
    if q_count == 1:
        return 0.1
    if q_count <= 3:
        return 0.4
    return 0.7


def _score_imperative_verbs(request: ParsedRequest) -> float:
    text = request.user_messages_text
    count = _count_matches(_IMPERATIVE_KEYWORDS, text)
    return min(count * 0.25, 1.0)


def _score_constraints(request: ParsedRequest) -> float:
    text = request.all_messages_text
    count = _count_matches(_CONSTRAINT_KEYWORDS, text)
    return min(count * 0.3, 1.0)


def _score_output_format(request: ParsedRequest) -> float:
    text = request.all_messages_text
    count = _count_matches(_OUTPUT_FORMAT_KEYWORDS, text)
    return min(count * 0.3, 1.0)


def _score_references(request: ParsedRequest) -> float:
    text = request.all_messages_text
    count = _count_matches(_REFERENCE_PATTERNS, text)
    return min(count * 0.3, 1.0)


def _score_negation(request: ParsedRequest) -> float:
    text = request.all_messages_text
    count = _count_matches(_NEGATION_PATTERNS, text)
    density = count / max(len(text.split()), 1)
    return min(density * 8.0, 1.0)


def _score_domain_specificity(request: ParsedRequest) -> float:
    text = request.all_messages_text
    count = _count_matches(_DOMAIN_KEYWORDS, text)
    return min(count * 0.25, 1.0)


def _score_agentic(request: ParsedRequest) -> float:
    """Detect agentic/tool-use patterns."""
    score = 0.0
    if request.has_tools:
        score += 0.6
    text = request.all_messages_text
    count = _count_matches(_AGENTIC_KEYWORDS, text)
    score += min(count * 0.2, 0.4)
    return min(score, 1.0)


# ── Dimension registry ───────────────────────────────────────────────

DIMENSION_SCORERS: dict[str, callable] = {
    "token_count": _score_token_count,
    "code_presence": _score_code_presence,
    "reasoning_markers": _score_reasoning_markers,
    "technical_terms": _score_technical_terms,
    "creative_markers": _score_creative_markers,
    "simple_indicators": _score_simple_indicators,
    "multi_step_patterns": _score_multi_step,
    "question_complexity": _score_question_complexity,
    "imperative_verbs": _score_imperative_verbs,
    "constraint_indicators": _score_constraints,
    "output_format": _score_output_format,
    "reference_complexity": _score_references,
    "negation_complexity": _score_negation,
    "domain_specificity": _score_domain_specificity,
    "agentic_task": _score_agentic,
}


def _sigmoid_confidence(score: float, boundaries: dict[str, float]) -> float:
    """Compute confidence using sigmoid on distance from nearest tier boundary.

    Higher distance from boundary → higher confidence.
    Close to boundary → ambiguous → low confidence.
    """
    boundary_values = sorted(boundaries.values())
    min_distance = float("inf")
    for b in boundary_values:
        dist = abs(score - b)
        min_distance = min(min_distance, dist)

    # Sigmoid: 1 / (1 + e^(-k*x + shift))
    # Calibrated so: distance=0 → ~0.38, distance=0.06 → ~0.7, distance=0.15 → ~0.96
    k = 25.0
    shift = 0.5
    return 1.0 / (1.0 + math.exp(-k * min_distance + shift))


def classify_fast_path(
    request: ParsedRequest,
    config: FastPathConfig,
) -> ClassificationResult | None:
    """Run the 14-dimension fast-path classifier.

    Returns ClassificationResult if confidence is above threshold,
    or None if the request is ambiguous and needs full classification.
    """
    if not config.enabled:
        return None

    # Score all dimensions
    scores: dict[str, float] = {}
    weighted_sum = 0.0

    for dim_name, scorer in DIMENSION_SCORERS.items():
        weight = config.weights.get(dim_name, 0.0)
        raw_score = scorer(request)
        scores[dim_name] = raw_score
        weighted_sum += weight * raw_score

    # Find dominant dimension
    dominant = max(scores, key=lambda k: abs(scores[k] * config.weights.get(k, 0)))

    # Override checks
    is_agentic = scores.get("agentic_task", 0) > 0.5

    # 2+ reasoning keywords in user prompt → force REASONING
    reasoning_score = scores.get("reasoning_markers", 0)
    if reasoning_score >= 1.0:
        return ClassificationResult(
            tier=ComplexityTier.REASONING,
            confidence=0.85,
            scores=scores,
            is_agentic=is_agentic,
            dominant_dimension="reasoning_markers",
        )

    # Tools present → boost toward COMPLEX/REASONING
    if request.has_tools:
        weighted_sum += 0.15

    # Map to tier
    boundaries = config.tier_boundaries
    if weighted_sum < boundaries.get("simple", 0.0):
        tier = ComplexityTier.SIMPLE
    elif weighted_sum < boundaries.get("medium", 0.3):
        tier = ComplexityTier.MEDIUM
    elif weighted_sum < boundaries.get("complex", 0.5):
        tier = ComplexityTier.COMPLEX
    else:
        tier = ComplexityTier.REASONING

    # Compute confidence
    confidence = _sigmoid_confidence(weighted_sum, boundaries)

    # Below threshold → ambiguous → return None for full classification
    if confidence < config.confidence_threshold:
        return None

    return ClassificationResult(
        tier=tier,
        confidence=confidence,
        scores=scores,
        is_agentic=is_agentic,
        dominant_dimension=dominant,
    )
