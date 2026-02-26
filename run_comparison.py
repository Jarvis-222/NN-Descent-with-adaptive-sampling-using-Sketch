#!/usr/bin/env python3
"""
Comparison benchmark: Optimized Vanilla vs Source-Based Adaptive NN-Descent.

Reports recall, timing, and sketch overhead using the datasketches library
for real Space-Saving sketch operations.

Usage:
    python run_comparison.py
"""

import numpy as np
import time
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.nn_descent_optimized import OptimizedNNDescent
from src.nn_descent_adaptive import AdaptiveNNDescent
from src.distance_metrics import euclidean_distance
from src.utils import brute_force_knn, compute_recall


# ── Data generation ──────────────────────────────────────────────────

def generate_clustered(n, d, n_clusters=10, cluster_std=0.3, seed=42):
    rng = np.random.default_rng(seed)
    pts_per = n // n_clusters
    data = []
    for _ in range(n_clusters):
        centre = rng.standard_normal(d) * 5
        data.extend(centre + rng.standard_normal((pts_per, d)) * cluster_std)
    # Handle remainder
    remaining = n - len(data)
    if remaining > 0:
        centre = rng.standard_normal(d) * 5
        data.extend(centre + rng.standard_normal((remaining, d)) * cluster_std)
    return np.array(data, dtype=np.float32)


def generate_uniform(n, d, seed=42):
    return np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)


# ── Ground truth ─────────────────────────────────────────────────────

def compute_ground_truth(data, k):
    """Use sklearn if available, otherwise brute-force."""
    try:
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(data)
        _, idx = nbrs.kneighbors(data)
        return idx[:, 1:]
    except ImportError:
        return brute_force_knn(data, k, euclidean_distance)


# ── Single comparison run ────────────────────────────────────────────

def run_comparison(n=1000, d=50, k=10, data_type="clustered",
                   rho=0.8, sample_rate=0.5, seed=42):

    print(f"\n{'='*70}")
    print(f"  {data_type.upper()} DATA  —  n={n}, d={d}, k={k}")
    print(f"  ρ={rho}, sample_rate={sample_rate}")
    print(f"{'='*70}")

    # Data
    if data_type == "clustered":
        data = generate_clustered(n, d, n_clusters=max(5, n // 500), seed=seed)
    else:
        data = generate_uniform(n, d, seed=seed)

    # Ground truth
    t0 = time.time()
    gt = compute_ground_truth(data, k)
    print(f"  Ground truth: {time.time() - t0:.2f}s\n")

    rows = []

    # ── Vanilla ───────────────────────────────────────────────────────
    print("  Running Vanilla NN-Descent ...")
    np.random.seed(seed)
    model = OptimizedNNDescent(k=k, rho=rho, max_iterations=20,
                                seed=seed, verbose=False)
    t0 = time.time()
    model.fit(data)
    vt = time.time() - t0
    vr = compute_recall(model.get_neighbors(), gt)
    print(f"    Time: {vt:.3f}s   Recall: {vr:.4f}")
    rows.append(("Vanilla (baseline)", vt, vr, ""))

    # ── Adaptive full ─────────────────────────────────────────────────
    print("  Running Adaptive (full eval + sketch overhead) ...")
    np.random.seed(seed)
    model = AdaptiveNNDescent(k=k, rho=rho, max_iterations=20,
                               warmup_iterations=2, sketch_lg_max_k=6,
                               sample_rate=1.0, seed=seed, verbose=False)
    t0 = time.time()
    model.fit(data)
    ft = time.time() - t0
    fr = compute_recall(model.get_neighbors(), gt)
    oh = (ft / vt - 1) * 100
    print(f"    Time: {ft:.3f}s   Recall: {fr:.4f}   Overhead: {oh:+.1f}%")
    rows.append(("Adaptive (full + sketch)", ft, fr, f"overhead {oh:+.1f}%"))

    # ── Adaptive sampled ──────────────────────────────────────────────
    tag = f"{sample_rate:.0%} sampled"
    print(f"  Running Adaptive ({tag}) ...")
    np.random.seed(seed)
    model = AdaptiveNNDescent(k=k, rho=rho, max_iterations=20,
                               warmup_iterations=2, sketch_lg_max_k=6,
                               sample_rate=sample_rate, seed=seed, verbose=False)
    t0 = time.time()
    model.fit(data)
    st = time.time() - t0
    sr = compute_recall(model.get_neighbors(), gt)
    sp = vt / st
    print(f"    Time: {st:.3f}s   Recall: {sr:.4f}   Speedup: {sp:.2f}x")
    rows.append((f"Adaptive ({tag})", st, sr, f"{sp:.2f}x speed"))

    # ── Summary table ─────────────────────────────────────────────────
    print(f"\n  {'─'*64}")
    print(f"  {'Method':<30} {'Time':<10} {'Recall':<10} {'Note':<16}")
    print(f"  {'─'*64}")
    for name, t, r, note in rows:
        print(f"  {name:<30} {t:<10.3f} {r:<10.4f} {note:<16}")
    print(f"  {'─'*64}")

    return rows


# ── Entry point ──────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "#" * 70)
    print("#  NN-Descent: Vanilla vs Source-Based Adaptive")
    print("#  Using datasketches library for sketch operations")
    print("#" * 70)

    run_comparison(n=1000, d=50, k=10, data_type="clustered", sample_rate=0.5)
    run_comparison(n=1000, d=50, k=10, data_type="uniform",   sample_rate=0.5)
    run_comparison(n=5000, d=50, k=20, data_type="clustered", sample_rate=0.5)
    run_comparison(n=5000, d=50, k=20, data_type="uniform",   sample_rate=0.5)
