# Two Sampling Mechanisms in NN-Descent

This document explains the two distinct sampling strategies used in our NN-Descent implementations, where they come from, and how they interact.

---

## Overview

```
Point v needs better neighbors. How do we find candidates?

Step 1: Neighbor list sampling (ρ)     → Controls candidate DISCOVERY
Step 2: Candidate evaluation sampling  → Controls candidate EVALUATION
         (sample_rate)
```

---

## 1. ρ-Sampling (Neighbor List Sampling)

### Origin

**This is from the original NN-Descent paper** (Dong et al., "Efficient K-Nearest Neighbor Graph Construction for Generic Similarity Measures", WWW 2011, Section 4.2).

The paper states:

> *"In each iteration, we sample ρK items from each of new[v] and old[v] where ρ is a parameter."*

The paper recommends ρ = 1.0 for best recall and ρ < 1.0 to trade recall for speed.

### What It Does

Each point v has two neighbor lists:
- **NEW[v]**: neighbors that were added since the last iteration
- **OLD[v]**: neighbors from previous iterations (already explored)

Before the local join step, ρ-sampling **caps** these lists:

```python
sample_size = int(ρ × k)

# If v has too many NEW neighbors, randomly keep only sample_size
if len(new_nbs) > sample_size:
    new_nbs = rng.choice(new_nbs, sample_size, replace=False)

# Same for OLD neighbors
if len(old_nbs) > sample_size:
    old_nbs = rng.choice(old_nbs, sample_size, replace=False)
```

### Why Lists Grow Beyond k

Neighbor lists can exceed k because of **reverse neighbors**. If point u has v as a neighbor, then u appears in v's reverse neighbor list and gets merged into v's NEW or OLD list. So even though v has at most k direct neighbors, the merged list can be much larger.

### Effect

| ρ value | sample_size (k=20) | Effect |
|---------|-------------------|--------|
| 1.0 | 20 | Keep all — maximum exploration, most distance computations |
| 0.8 | 16 | Keep most — good balance |
| 0.5 | 10 | Keep half — faster but fewer candidates discovered |

### Code Location

- [nn_descent_optimized.py](file:///home/jainish/UWaterloo/Winter_2026/NN-Descent-with-adaptive-sampling-using-Sketch/src/nn_descent_optimized.py#L82-L85): Lines 82-85
- [nn_descent_adaptive.py](file:///home/jainish/UWaterloo/Winter_2026/NN-Descent-with-adaptive-sampling-using-Sketch/src/nn_descent_adaptive.py#L106-L109): Lines 106-109

---

## 2. Candidate Evaluation Sampling (sample_rate)

### Origin

**This is our contribution** — not in the original paper. The original NN-Descent evaluates ALL discovered candidates. We introduce a second sampling step that selects a fraction of candidates for evaluation.

### What It Does

After the local join discovers candidates (neighbors-of-neighbors), we have a candidate pool for each point. Instead of evaluating all of them:

```python
# Pool of candidates discovered for point v
candidates = [c1, c2, c3, ..., c200]  # e.g., 200 candidates

budget = int(len(candidates) × sample_rate)  # e.g., 100

# RANDOM (optimized): pick uniformly at random
sampled = rng.choice(candidates, budget)

# ADAPTIVE: pick weighted by sketch scores
sampled = tracker.sample_candidates(v, candidates, ...)
```

### Two Strategies

#### Random Sampling (Optimized)
Every candidate has **equal probability** of being selected. Simple, unbiased, but wastes budget on unpromising candidates.

#### Sketch-Weighted Sampling (Adaptive)
Candidates from historically **productive sources** get higher selection probability:

```
weight(candidate) = base_weight + Σ freq(source) × source_boost
```

Where:
- `base_weight = 1.0` — ensures every candidate has a chance (exploration)
- `source_boost = 0.5` — how much each past success increases weight (exploitation)
- `freq(source)` — how many times this source led to successful updates (from the Space-Saving sketch)

### Effect

| sample_rate | Candidates evaluated | Computation savings |
|-------------|---------------------|-------------------|
| 1.0 | All | None (baseline) |
| 0.5 | Half | ~50% fewer distance computations |
| 0.3 | 30% | ~70% fewer distance computations |

### Code Location

- Optimized (random): [nn_descent_optimized.py](file:///home/jainish/UWaterloo/Winter_2026/NN-Descent-with-adaptive-sampling-using-Sketch/src/nn_descent_optimized.py#L108-L113): Lines 108-113
- Adaptive (sketch): [nn_descent_adaptive.py](file:///home/jainish/UWaterloo/Winter_2026/NN-Descent-with-adaptive-sampling-using-Sketch/src/nn_descent_adaptive.py#L138-L153): Lines 138-153

---

## How They Interact

```
                    ρ-sampling                    sample_rate
                    (neighbor lists)              (candidate pool)
                         │                              │
   NEW[v] = [a,b,c,d,e] │ ρ=0.5,k=10 → [a,c,e]       │
   OLD[v] = [f,g,h,i,j] │ ρ=0.5,k=10 → [f,h,j]       │
                         │                              │
                         ▼                              │
              ┌─────────────────────┐                   │
              │   LOCAL JOIN        │                   │
              │   Explore neighbors │                   │
              │   of [a,c,e,f,h,j] │                   │
              └────────┬────────────┘                   │
                       │                                │
              Candidates: [w1,w2,...,w80]                │
                       │                                │
                       ▼                                ▼
              ┌─────────────────────────────────────────────┐
              │ sample_rate = 0.5 → evaluate 40 candidates  │
              │                                             │
              │  Random: pick 40 uniformly                  │
              │  Adaptive: pick 40 weighted by sketch       │
              └──────────────┬──────────────────────────────┘
                             │
                    Compute distances
                    Update graph
```

### Key Insight

- **ρ controls the input** — how many neighbors participate in candidate discovery
- **sample_rate controls the output** — how many discovered candidates get evaluated
- Higher ρ → bigger candidate pool → more room for smart selection → adaptive benefits more

---

## Warmup Phase

During warmup (first 2 iterations), both random and adaptive use **random sampling at the sample_rate**. This ensures:

1. **Fair comparison**: Both strategies have identical computation budget during warmup
2. **Sketch training**: Random successes during warmup provide initial frequency data for the sketch
3. **Post-warmup**: Adaptive switches to sketch-weighted sampling; random continues uniform sampling

---

## Summary Table

| | ρ-sampling | sample_rate |
|---|---|---|
| **From** | NN-Descent paper (Dong 2011) | Our contribution |
| **Controls** | Neighbor list size | Candidate evaluation budget |
| **When** | Before local join | After candidate discovery |
| **Method** | Always random | Random (optimized) or sketch-weighted (adaptive) |
| **Default** | 0.5 in main.py, 0.8 in benchmark | 1.0 (evaluate all) |
| **Trade-off** | Fewer candidates discovered | Fewer candidates evaluated |
