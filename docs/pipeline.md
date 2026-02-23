# Pipeline Deep Dive

Detailed walkthrough of the 9-stage request pipeline and the 15-dimension fast-path classifier.

---

## The 9-Stage Request Pipeline

```
Request In ──▶ ① Parse ──▶ ② Dedup ──▶ ③ Session ──▶ ④ Classify
                                                          │
              ⑨ Post-process ◀── ⑧ Fallback ◀── ⑦ Route ◀── ⑥ Compress ◀── ⑤ Decide
```

| Stage | What Happens | Origin |
|-------|-------------|--------|
| **① Parse** | Extract messages, tools, model, max_tokens from OpenAI format | Semantic Router |
| **② Dedup** | SHA-256 hash after canonicalization; return cached if hit | ClawRouter |
| **③ Session** | Fingerprint conversation; check for existing model pin | ClawRouter |
| **④ Classify** | 15-dimension fast-path scorer (< 1 ms); semantic embedding fallback for ambiguous cases | ClawRouter + Semantic Router |
| **⑤ Decide** | Map tier → model, estimate cost, check context window fit | Semantic Router |
| **⑥ Compress** | If > 180 KB: whitespace, dedup, JSON compaction | ClawRouter |
| **⑦ Route** | Forward to selected provider (vLLM or Gemini) | Semantic Router |
| **⑧ Fallback** | On error/429/timeout: try fallback chain, then degradation model | ClawRouter |
| **⑨ Post-process** | Update session pin, dedup cache, metrics; add response headers | Both |

## Fast-Path Classifier: The 15 Dimensions

The fast-path is the core innovation from ClawRouter — a regex-based scorer that avoids neural inference for clear-cut requests.

### How it works

Each dimension scores the request on a scale of roughly -1.0 to +1.0. Scores are multiplied by their weight and summed. The weighted sum maps to a complexity tier:

```
Input: "What is a Python decorator?"

  Dimension              Score    Weight   Contribution
  ─────────────────────  ──────   ──────   ────────────
  reasoning_markers       0.00    × 0.18   =  0.000
  code_presence           0.00    × 0.15   =  0.000
  multi_step_patterns     0.00    × 0.12   =  0.000
  technical_terms         0.20    × 0.10   =  0.020
  token_count            -0.50    × 0.08   = -0.040   ◀ Short input
  simple_indicators      -1.00    × 0.08   = -0.080   ◀ "What is" pattern
  ...
  ─────────────────────────────────────────────────────
  Weighted sum:                             = -0.067
  Nearest boundary:                           0.0 (SIMPLE/MEDIUM)
  Distance:                                   0.067
  Sigmoid confidence:                         0.76   ◀ Above 0.7 threshold
  Result:                                     SIMPLE ✓
```

### Confidence gating

When the weighted sum falls near a tier boundary, the sigmoid confidence drops below the threshold (0.7 by default) and the request is escalated to the **semantic embedding classifier**.

### Semantic Embedding Fallback

When the fast-path is ambiguous (~14% of requests), the semantic classifier provides true meaning-based classification:

```
Ambiguous request → Embed with all-MiniLM-L6-v2 (22M params)
                      → Cosine similarity to tier anchor prompts
                      → Mean-of-top-k per tier
                      → Highest scoring tier wins
```

The classifier uses ~6-7 pre-defined anchor prompts per tier that represent typical requests at each complexity level. New requests are embedded and compared by cosine similarity — requests semantically similar to "Prove by induction..." will route to REASONING even without the keyword "prove".

**Latency**: ~5-20ms on CPU (vs. 40μs for fast-path). Only fires for ambiguous cases.

**Graceful degradation**: If `sentence-transformers` is not installed, the router silently falls back to heuristic re-scoring. Install with: `pip install semantic-claw-router[semantic]`

### Reasoning override

If the user message contains 2+ reasoning keywords ("prove", "derive", "induction", "theorem", "contradiction", etc.), the classifier forces the REASONING tier at 85% confidence, regardless of the overall weighted sum.

### Tier boundaries

| Tier | Score Range | Typical Requests |
|------|------------|-----------------|
| SIMPLE | < 0.0 | Definitions, lookups, greetings |
| MEDIUM | 0.0 – 0.3 | Code generation, debugging, documentation |
| COMPLEX | 0.3 – 0.5 | Multi-file refactoring, system design |
| REASONING | > 0.5 (or override) | Proofs, formal analysis, complex debugging |

### The 15 dimensions

| Dimension | Weight | Positive Signal | Negative Signal |
|-----------|--------|----------------|-----------------|
| reasoning_markers | 0.18 | "prove", "derive", "induction", "theorem" | — |
| code_presence | 0.15 | Code blocks, `def`, `class`, `import` | — |
| multi_step_patterns | 0.12 | "first...then", "step 1", sequential instructions | — |
| technical_terms | 0.10 | "algorithm", "kubernetes", "distributed" | — |
| token_count | 0.08 | Long input (>2000 tokens) | Short input (<100 tokens) |
| simple_indicators | 0.08 | — | "what is", "define", "translate" |
| creative_markers | 0.05 | "write a story", "compose", "brainstorm" | — |
| question_complexity | 0.05 | Multiple question marks, nested conditions | — |
| constraint_indicators | 0.04 | "must", "ensure", "require" | — |
| agentic_task | 0.04 | `tools[]` array, file ops, "try again" | — |
| imperative_verbs | 0.03 | "implement", "design", "build", "refactor" | — |
| output_format | 0.03 | "return as JSON", "as a table", "YAML" | — |
| reference_complexity | 0.02 | Cross-references, citations | — |
| domain_specificity | 0.02 | Medical, legal, scientific vocabulary | — |
| negation_complexity | 0.01 | Double negation, complex constraints | — |
