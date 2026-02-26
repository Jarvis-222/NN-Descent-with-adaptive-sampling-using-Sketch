# Adaptive NN-Descent Experiments
#
# Comparing:
# 1. Naive (Full scan)
# 2. Optimized (Local Join + Incremental)
# 3. Adaptive (Sketch-based sampling)

import argparse
import time
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_generator import generate_uniform, generate_gaussian, generate_high_hubness, load_sift
from src.distance_metrics import get_distance_function
from src.nn_descent_naive import NaiveNNDescent
from src.nn_descent_optimized import OptimizedNNDescent
from src.nn_descent_adaptive import AdaptiveNNDescent
from src.utils import brute_force_knn, compute_recall


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Dataset settings
    parser.add_argument('--data', type=str, default='random_uniform',
                        choices=['random_uniform', 'random_gaussian', 'clustered', 'sift', 'high_hubness'])
    parser.add_argument('--sift-path', type=str,
                        default='data/siftsmall/siftsmall_base.fvecs',
                        help='Path to SIFT .fvecs file')
    parser.add_argument('--n', type=int, default=500)
    parser.add_argument('--dim', type=int, default=50)
    parser.add_argument('--n-clusters', type=int, default=10)

    # Algorithm settings
    parser.add_argument('--algorithm', type=str, default='all',
                        choices=['all', 'naive', 'optimized', 'adaptive'])
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--distance', type=str, default='euclidean',
                        choices=['euclidean', 'cosine', 'manhattan'])

    # Common params
    parser.add_argument('--max-iter', type=int, default=20)
    parser.add_argument('--rho', type=float, default=0.8)
    parser.add_argument('--delta', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quiet', action='store_true')

    # Adaptive specific
    parser.add_argument('--sketch-size', type=int, default=6)
    parser.add_argument('--sample-rate', type=float, default=1.0,
                        help='Candidate sample rate in adaptive mode')
    parser.add_argument('--warmup', type=int, default=2,
                        help='Warmup iterations before adaptive sampling')

    args = parser.parse_args()
    verbose = not args.quiet

    # ── Generate data ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"NN-Descent Algorithm Comparison")
    print(f"{'='*60}")

    if args.data == 'sift':
        data = load_sift(args.sift_path, n=args.n)
        args.dim = data.shape[1]
    elif args.data == 'high_hubness':
        data = generate_high_hubness(args.n, args.dim, seed=args.seed)
    elif args.data == 'random_gaussian':
        data = generate_gaussian(args.n, args.dim, seed=args.seed)
    else:
        data = generate_uniform(args.n, args.dim, seed=args.seed)

    distance_fn = get_distance_function(args.distance)

    print(f"Data: {args.data}, n={args.n}, dim={args.dim}")
    print(f"Distance: {args.distance}, k={args.k}")

    # ── Compute ground truth ──────────────────────────────────────────
    print(f"\nComputing ground truth (sklearn brute force)...")
    from sklearn.neighbors import NearestNeighbors
    t0 = time.time()
    nbrs = NearestNeighbors(n_neighbors=args.k + 1, algorithm='brute', metric='euclidean').fit(data)
    _, ground_truth = nbrs.kneighbors(data)
    ground_truth = ground_truth[:, 1:]  # remove self
    gt_time = time.time() - t0
    print(f"  Done in {gt_time:.2f}s")

    results = {}
    step = 1

    # ── Naive ─────────────────────────────────────────────────────────
    if args.algorithm in ['all', 'naive']:
        print(f"\n[{step}] Naive NN-Descent")
        model = NaiveNNDescent(
            k=args.k, distance_fn=distance_fn,
            max_iterations=args.max_iter, seed=args.seed, verbose=verbose,
        )
        t0 = time.time()
        model.fit(data)
        elapsed = time.time() - t0
        recall = compute_recall(model.get_neighbors(), ground_truth)
        results['naive'] = {'time': elapsed, 'recall': recall}
        print(f"  Time: {elapsed:.3f}s | Recall: {recall:.4f}")
        step += 1

    # ── Optimized ─────────────────────────────────────────────────────
    if args.algorithm in ['all', 'optimized']:
        print(f"\n[{step}] Optimized NN-Descent (ρ={args.rho}, δ={args.delta})")
        model = OptimizedNNDescent(
            k=args.k, distance_fn=distance_fn,
            rho=args.rho, delta=args.delta,
            max_iterations=args.max_iter,
            sample_rate=args.sample_rate,
            seed=args.seed, verbose=verbose,
        )
        t0 = time.time()
        model.fit(data)
        elapsed = time.time() - t0
        recall = compute_recall(model.get_neighbors(), ground_truth)
        results['optimized'] = {'time': elapsed, 'recall': recall}
        print(f"  Time: {elapsed:.3f}s | Recall: {recall:.4f}")
        step += 1

    # ── Adaptive ──────────────────────────────────────────────────────
    if args.algorithm in ['all', 'adaptive']:
        print(f"\n[{step}] Adaptive NN-Descent "
              f"(sketch=2^{args.sketch_size}, warmup={args.warmup}, "
              f"sample_rate={args.sample_rate})")
        model = AdaptiveNNDescent(
            k=args.k, distance_fn=distance_fn,
            rho=args.rho, delta=args.delta,
            max_iterations=args.max_iter,
            warmup_iterations=args.warmup,
            sketch_lg_max_k=args.sketch_size,
            sample_rate=args.sample_rate,
            seed=args.seed, verbose=verbose,
        )
        t0 = time.time()
        model.fit(data)
        elapsed = time.time() - t0
        recall = compute_recall(model.get_neighbors(), ground_truth)
        results['adaptive'] = {'time': elapsed, 'recall': recall}
        print(f"  Time: {elapsed:.3f}s | Recall: {recall:.4f}")
        step += 1

    # ── Summary ───────────────────────────────────────────────────────
    if len(results) > 1:
        print(f"\n{'─'*60}")
        print(f"{'Method':<25} {'Time (s)':<12} {'Recall':<10}")
        print(f"{'─'*60}")
        for name, r in results.items():
            print(f"{name:<25} {r['time']:<12.3f} {r['recall']:<10.4f}")
        print(f"{'─'*60}")


if __name__ == '__main__':
    main()
