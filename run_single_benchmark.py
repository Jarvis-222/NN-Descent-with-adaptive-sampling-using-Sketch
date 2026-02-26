#!/usr/bin/env python3
"""Run a single benchmark. Called by other scripts."""

import numpy as np, sys, os, time, json
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.knn_graph import KNNGraph
from src.distance_metrics import euclidean_distance
from src.source_frequency_tracker import SourceFrequencyTracker
from src.utils import compute_recall


def run_config(data, gt, k, sample_rate, mode, rho=0.8, seed=42):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    n = len(data)
    delta = 0.001
    sample_size = max(1, int(rho * k))
    warmup = 2
    threshold = delta * n * k

    graph = KNNGraph(n, k)
    graph.initialize_random(exclude_self=True)
    graph.update_distances(data, euclidean_distance)

    tracker = SourceFrequencyTracker(n, lg_max_k=6) if mode == 'adaptive' else None
    total_dist = 0
    iter_log = []

    for iteration in range(20):
        updates = 0
        iter_dist = 0
        is_warmup = iteration < warmup
        t0 = time.time()

        reverse = graph.get_reverse_neighbors()
        new_lists, old_lists = [], []
        for v in range(n):
            new_nbs = [nb.index for nb in graph.get_new_neighbors(v)]
            old_nbs = [nb.index for nb in graph.get_old_neighbors(v)]
            for u in reverse[v]:
                is_new = any(nb.index == v for nb in graph.get_new_neighbors(u))
                target = new_nbs if is_new else old_nbs
                if u not in target: target.append(u)
            if len(new_nbs) > sample_size:
                new_nbs = list(rng.choice(new_nbs, sample_size, replace=False))
            if len(old_nbs) > sample_size:
                old_nbs = list(rng.choice(old_nbs, sample_size, replace=False))
            new_lists.append(new_nbs)
            old_lists.append(old_nbs)

        new_sets = [set(new_lists[i]) for i in range(n)]
        graph.mark_all_old()

        for v in range(n):
            current_nbs = set(graph.get_neighbor_indices(v))
            csrc = defaultdict(list)
            for u in new_lists[v]:
                for nb in graph.get_neighbors(u):
                    w = nb.index
                    if w != v and w not in current_nbs:
                        csrc[w].append(u)
            for u in old_lists[v]:
                for w in new_sets[u]:
                    if w != v and w not in current_nbs:
                        csrc[w].append(u)
            if not csrc: continue
            cands = list(csrc.keys())

            if sample_rate >= 1.0 or len(cands) <= 1:
                sampled = cands
            elif is_warmup or mode == 'random':
                # Warmup + random mode: sample at the rate
                budget = max(1, int(len(cands) * sample_rate))
                idx = rng.choice(len(cands), size=budget, replace=False)
                sampled = [cands[i] for i in idx]
            else:
                # Adaptive: sketch-weighted sampling
                budget = max(1, int(len(cands) * sample_rate))
                sampled = tracker.sample_candidates(v, cands, csrc, budget, rng)

            for c in sampled:
                dist = euclidean_distance(data[v], data[c])
                iter_dist += 1

                evicted_v = graph.try_update(v, c, dist)
                if evicted_v is not None:
                    updates += 1
                    if tracker:
                        src = csrc[c][0] if csrc[c] else c
                        tracker.record_source_success(v, src)

                if graph.try_update(c, v, dist) is not None:
                    updates += 1

        total_dist += iter_dist
        elapsed = time.time() - t0
        iter_log.append({"iter": iteration+1, "updates": updates, "dist": iter_dist, "time": round(elapsed, 1)})

        if updates < threshold:
            break

    recall = compute_recall(graph.to_array(), gt)
    return {"recall": round(recall, 4), "dist_comps": total_dist, "iterations": len(iter_log), "iter_log": iter_log}


if __name__ == '__main__':
    mode = sys.argv[1]       # 'random' or 'adaptive'
    rate = float(sys.argv[2]) # e.g. 0.5
    rho = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8  # e.g. 1.0

    K = 20

    # ── Load SIFT-10K ────────────────────────────────────────────
    fvecs_path = 'data/siftsmall/siftsmall_base.fvecs'
    cache_path = 'data/cache/sift10k_data.npy'

    if os.path.exists(fvecs_path):
        print(f'Loading SIFT from {fvecs_path}...', flush=True)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from src.data_generator import load_sift
        data = load_sift(fvecs_path, n=10000)
    elif os.path.exists(cache_path):
        print(f'Loading SIFT from {cache_path}...', flush=True)
        data = np.load(cache_path)
    else:
        raise FileNotFoundError(f'No SIFT data found at {fvecs_path} or {cache_path}')

    print(f'  Data shape: {data.shape}', flush=True)

    # ── Compute ground truth from scratch ────────────────────────
    print(f'Computing ground truth (sklearn brute force, k={K})...', flush=True)
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=K + 1, algorithm='brute', metric='euclidean').fit(data)
    _, gt = nbrs.kneighbors(data)
    gt = gt[:, 1:]  # remove self
    print(f'  Ground truth shape: {gt.shape}', flush=True)

    # ── Run benchmark ────────────────────────────────────────────
    label = f'{mode}_{rate:.0%}_rho{rho}'
    print(f'[{label}] Starting (rho={rho})...', flush=True)

    t0 = time.time()
    result = run_config(data, gt, k=K, sample_rate=rate, mode=mode, rho=rho)
    result['time'] = round(time.time() - t0, 1)
    result['mode'] = mode
    result['rate'] = rate
    result['rho'] = rho

    os.makedirs('data/cache', exist_ok=True)
    with open(f'data/cache/result_{mode}_{int(rate*100)}.json', 'w') as f:
        json.dump(result, f, indent=2)

    print(f'[{label}] Done: recall={result["recall"]:.4f}, dist={result["dist_comps"]:,d}, '
          f'iters={result["iterations"]}, time={result["time"]:.1f}s', flush=True)

