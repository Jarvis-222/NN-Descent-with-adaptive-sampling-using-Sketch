# NN-Descent with Adaptive Sampling using Space-Saving Sketch

This repository implements the **NN-Descent** algorithm for approximate k-nearest neighbor (k-NN) graph construction, along with an adaptive sampling extension that uses **Space-Saving sketches** (via the [Apache DataSketches](https://datasketches.apache.org/) library) to prioritise promising candidates during graph refinement.

## Overview

NN-Descent iteratively improves a k-NN graph by exploring neighbours-of-neighbours as potential closer neighbours. Our adaptive variant tracks, for each vertex, which *source* neighbours most frequently lead to successful graph updates. Candidates discovered through productive sources receive higher evaluation priority, allowing us to reduce distance computations without sacrificing recall.

### Algorithms implemented

| Variant | Description |
|---|---|
| **Naive** | Baseline — evaluates all neighbour-of-neighbour pairs |
| **Optimized** | Local join + incremental search + sampling + early termination |
| **Adaptive** | Optimized + per-vertex Space-Saving sketch for source-based weighted candidate sampling |

## Project Structure

```
├── src/
│   ├── distance_metrics.py          # Euclidean, cosine, Manhattan
│   ├── knn_graph.py                 # k-NN graph data structure
│   ├── nn_descent_naive.py          # Baseline NN-Descent
│   ├── nn_descent_optimized.py      # Optimized with local join
│   ├── nn_descent_adaptive.py       # Adaptive with sketch sampling
│   ├── source_frequency_tracker.py  # Per-vertex Space-Saving sketch
│   ├── data_generator.py            # Synthetic data generators + SIFT loader
│   └── utils.py                     # Brute-force k-NN, recall, timer
├── tests/
│   ├── test_nn_descent.py           # Tests for naive & optimized variants
│   ├── test_adaptive.py             # Tests for adaptive variant & tracker
│   └── test_data_generator.py       # Tests for data generators
├── main.py                          # CLI entry point for experiments
├── run_comparison.py                # Benchmark: vanilla vs adaptive
└── requirements.txt
```

## Getting Started

### Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running experiments

```bash
# Compare all three algorithms on clustered data
python main.py --algorithm all --data clustered --n 500 --k 10

# Run adaptive only with 50% candidate sampling
python main.py --algorithm adaptive --sample-rate 0.5 --n 1000 --k 20

# Run the full benchmark comparison
python run_comparison.py
```

### Running tests

```bash
pytest tests/ -v
```

## Key Parameters

| Parameter | Flag | Default | Description |
|---|---|---|---|
| `k` | `--k` | 10 | Number of nearest neighbours |
| `ρ` (rho) | `--rho` | 0.5 | Neighbour sampling rate |
| `δ` (delta) | `--delta` | 0.001 | Early termination threshold |
| Sketch size | `--sketch-size` | 6 | Log₂ of Space-Saving sketch capacity per vertex |
| Sample rate | `--sample-rate` | 1.0 | Fraction of candidates to evaluate (adaptive mode) |
| Warmup | `--warmup` | 2 | Iterations before enabling adaptive sampling |

## References

1. W. Dong, C. Moses, K. Li. "Efficient K-Nearest Neighbor Graph Construction for Generic Similarity Measures." *WWW 2011*.
2. A. Metwally, D. Agrawal, A. El Abbadi. "Efficient Computation of Frequent and Top-k Elements in Data Streams." *ICDT 2005*.
3. Apache DataSketches — https://datasketches.apache.org/
