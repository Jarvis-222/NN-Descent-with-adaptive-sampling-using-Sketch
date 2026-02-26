#!/usr/bin/env python3
"""
Quick benchmark: Random vs Adaptive on SIFT-5K and Gaussian-100d (high hubness).
Uses n=5000 for faster turnaround.
"""
import numpy as np, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_single_benchmark import run_config
from src.distance_metrics import euclidean_distance

K = 10
RATE = 0.5
SEED = 42
N = 5000

def p(msg): print(msg, flush=True)

def get_gt(data, k):
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(data)
    _, idx = nbrs.kneighbors(data)
    return idx[:, 1:]

def run_one(name, data, gt, k, rate, seed):
    p(f"\n{'='*60}")
    p(f"  {name}  |  n={len(data)}, d={data.shape[1]}, k={k}, rate={rate}")
    p(f"{'='*60}")

    p(f"  [RANDOM {rate:.0%}]...")
    t0 = time.time()
    rr = run_config(data, gt, k=k, sample_rate=rate, mode='random', seed=seed)
    p(f"    Recall={rr['recall']:.4f}  Dist={rr['dist_comps']:,d}  Iters={rr['iterations']}  T={time.time()-t0:.1f}s")

    p(f"  [ADAPTIVE {rate:.0%}]...")
    t0 = time.time()
    ra = run_config(data, gt, k=k, sample_rate=rate, mode='adaptive', seed=seed)
    p(f"    Recall={ra['recall']:.4f}  Dist={ra['dist_comps']:,d}  Iters={ra['iterations']}  T={time.time()-t0:.1f}s")

    diff = ra['recall'] - rr['recall']
    p(f"  --> Adaptive: {diff:+.4f} ({diff*100:+.2f} pp)")
    return rr, ra

results = []

# ── SIFT (first 5000 points) ─────────────────────────────────────
p("Loading SIFT (first 5000 pts)...")
sift_all = np.load('data/cache/sift10k_data.npy')[:N]
sift_gt = get_gt(sift_all, K)
rr, ra = run_one("SIFT-5K (d=128)", sift_all, sift_gt, K, RATE, SEED)
results.append(("SIFT-5K (d=128)", rr, ra))

# ── High-Hubness: Gaussian d=100 ─────────────────────────────────
p("\nGenerating Gaussian d=100 (HIGH HUBNESS)...")
gauss = np.random.default_rng(SEED).standard_normal((N, 100)).astype(np.float32)
gauss_gt = get_gt(gauss, K)
rr, ra = run_one("Gaussian-100d (HIGH HUBNESS)", gauss, gauss_gt, K, RATE, SEED)
results.append(("Gaussian-100d (high hub)", rr, ra))

# ── Low-hubness control: Gaussian d=10 ───────────────────────────
p("\nGenerating Gaussian d=10 (low hubness control)...")
gauss_low = np.random.default_rng(SEED).standard_normal((N, 10)).astype(np.float32)
gauss_low_gt = get_gt(gauss_low, K)
rr, ra = run_one("Gaussian-10d (low hubness)", gauss_low, gauss_low_gt, K, RATE, SEED)
results.append(("Gaussian-10d (low hub)", rr, ra))

# ── Summary ──────────────────────────────────────────────────────
p(f"\n\n{'='*70}")
p(f"  SUMMARY: k={K}, sample_rate={RATE}, n={N}")
p(f"{'='*70}")
p(f"{'Dataset':<30} {'Random':>10} {'Adaptive':>10} {'Δ (pp)':>10}")
p(f"{'-'*70}")
for name, rr, ra in results:
    d = (ra['recall'] - rr['recall']) * 100
    p(f"{name:<30} {rr['recall']:>10.4f} {ra['recall']:>10.4f} {d:>+10.2f}")
p(f"{'-'*70}")
