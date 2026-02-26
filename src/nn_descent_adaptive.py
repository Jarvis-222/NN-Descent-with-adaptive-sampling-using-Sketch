"""
Adaptive NN-Descent with Source-Based Sketch Sampling.

Extends the optimized NN-Descent algorithm with an adaptive candidate
sampling strategy. Instead of evaluating all discovered candidates
equally, this variant uses per-vertex Space-Saving sketches (by using the
datasketches library from) to track which source neighbors are most
productive. Candidates from productive sources receive higher
evaluation priority.

Algorithm outline:
  1. Initialise random k-NN graph.
  2. Warmup phase — evaluate ALL candidates; record source successes.
  3. Adaptive phase — weight candidates by source success, sample a
     fraction of them, evaluate, and continue recording successes.
  4. Terminate when convergence criterion is met.

The hypothesis is that many real-world datasets have a non-uniform
neighbourhood structure, so some sources are persistently more valuable
than others. By focusing evaluation budget on candidates from those
sources we can reduce the number of distance computations while
maintaining comparable recall.

Reference:
    - Dong et al., "Efficient K-Nearest Neighbor Graph Construction for
      Generic Similarity Measures", WWW 2011.
    - Metwally et al., "Efficient Computation of Frequent and Top-k
      Elements in Data Streams", ICDT 2005.
"""

import numpy as np
from collections import defaultdict
from typing import Callable, Dict, List, Optional
from .knn_graph import KNNGraph
from .source_frequency_tracker import SourceFrequencyTracker


def nn_descent_adaptive(
    data: np.ndarray,
    k: int,
    distance_fn: Callable,
    rho: float = 0.5,
    delta: float = 0.001,
    max_iterations: int = 20,
    warmup_iterations: int = 2,
    sketch_lg_max_k: int = 6,
    sample_rate: float = 1.0,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> KNNGraph:
    """
    Adaptive NN-Descent.
    
    Similar to the optimized version, but we sample candidates smartly.
    Instead of random sampling, we use the tracker to prioritize candidates
    from sources that have been successful in the past.
    """
    rng = np.random.default_rng(seed)
    if seed is not None:
        np.random.seed(seed)

    n = len(data)
    sample_size = max(1, int(rho * k))
    termination_threshold = delta * n * k

    if verbose:
        print(f"Adaptive NN-Descent: n={n}, k={k}, ρ={rho}, δ={delta}")
        print(f"  Sample size (neighbors): {sample_size}, "
              f"Warmup: {warmup_iterations} iters")
        print(f"  Sketch size: 2^{sketch_lg_max_k} = "
              f"{2 ** sketch_lg_max_k} items/vertex")
        print(f"  Candidate sample rate: {sample_rate:.0%}")

    # ── Initialization ────────────────────────────────────────────────
    graph = KNNGraph(n, k)
    graph.initialize_random(exclude_self=True)
    graph.update_distances(data, distance_fn)

    tracker = SourceFrequencyTracker(n, lg_max_k=sketch_lg_max_k)
    distance_computations = 0

    # ── Main loop ─────────────────────────────────────────────────────
    for iteration in range(max_iterations):
        updates = 0
        is_warmup = iteration < warmup_iterations

        reverse = graph.get_reverse_neighbors()

        # Build NEW / OLD lists (same as optimized variant)
        new_lists, old_lists = [], []
        for v in range(n):
            new_nbs = [nb.index for nb in graph.get_new_neighbors(v)]
            old_nbs = [nb.index for nb in graph.get_old_neighbors(v)]

            for u in reverse[v]:
                is_new_rev = any(
                    nb.index == v for nb in graph.get_new_neighbors(u)
                )
                target = new_nbs if is_new_rev else old_nbs
                if u not in target:
                    target.append(u)

            # ── ρ-sampling: keep sample_size neighbors ─────────────────
            if is_warmup:
                # Warmup: random ρ-sampling (no sketch data yet)
                if len(new_nbs) > sample_size:
                    new_nbs = list(rng.choice(new_nbs, sample_size, replace=False))
                if len(old_nbs) > sample_size:
                    old_nbs = list(rng.choice(old_nbs, sample_size, replace=False))
            else:
                # Adaptive: sketch-weighted ρ-sampling
                if len(new_nbs) > sample_size:
                    new_nbs = tracker.sample_neighbors(
                        v, new_nbs, sample_size, rng=rng)
                if len(old_nbs) > sample_size:
                    old_nbs = tracker.sample_neighbors(
                        v, old_nbs, sample_size, rng=rng)

            new_lists.append(new_nbs)
            old_lists.append(old_nbs)

        new_sets = [set(new_lists[i]) for i in range(n)]
        graph.mark_all_old()

        # ── Local Join with source attribution ────────────────────────
        for v in range(n):
            current_nbs = set(graph.get_neighbor_indices(v))
            candidate_sources: Dict[int, List[int]] = defaultdict(list)

            # NEW neighbours → explore ALL their neighbours
            for u in new_lists[v]:
                for nb in graph.get_neighbors(u):
                    w = nb.index
                    if w != v and w not in current_nbs:
                        candidate_sources[w].append(u)

            # OLD neighbours → explore only their NEW neighbours
            for u in old_lists[v]:
                for w in new_sets[u]:
                    if w != v and w not in current_nbs:
                        candidate_sources[w].append(u)

            if not candidate_sources:
                continue

            # ── Evaluate ALL candidates ───────────────────────────────
            for c, sources in candidate_sources.items():
                dist = distance_fn(data[v], data[c])
                distance_computations += 1

                # Try updating v's list with candidate c
                evicted_v = graph.try_update(v, c, dist)
                if evicted_v is not None:
                    updates += 1
                    # Record the source that led to this success
                    src = sources[0] if sources else c
                    tracker.record_source_success(v, src)

                # Try the reverse update: c's list with v
                evicted_c = graph.try_update(c, v, dist)
                if evicted_c is not None:
                    updates += 1

        phase = "warmup" if is_warmup else "adaptive"
        if verbose:
            print(f"  Iteration {iteration + 1} ({phase}): {updates} updates")

        if updates < termination_threshold:
            if verbose:
                print(f"  Early termination at iteration {iteration + 1} "
                      f"(updates {updates} < {termination_threshold:.0f})")
            break

    if verbose:
        print(f"  Total distance computations: {distance_computations}")
        print(f"  Total source successes tracked: {tracker.total_updates}")

    return graph


class AdaptiveNNDescent:
    """Wrapper with fit / get_neighbors interface."""

    def __init__(
        self,
        k: int = 10,
        distance_fn: Callable = None,
        rho: float = 0.5,
        delta: float = 0.001,
        max_iterations: int = 20,
        warmup_iterations: int = 2,
        sketch_lg_max_k: int = 6,
        sample_rate: float = 1.0,
        seed: Optional[int] = None,
        verbose: bool = True,
    ):
        self.k = k
        self.distance_fn = distance_fn
        self.rho = rho
        self.delta = delta
        self.max_iterations = max_iterations
        self.warmup_iterations = warmup_iterations
        self.sketch_lg_max_k = sketch_lg_max_k
        self.sample_rate = sample_rate
        self.seed = seed
        self.verbose = verbose
        self.graph_: Optional[KNNGraph] = None

    def fit(self, data: np.ndarray) -> "AdaptiveNNDescent":
        if self.distance_fn is None:
            from .distance_metrics import euclidean_distance
            self.distance_fn = euclidean_distance

        self.graph_ = nn_descent_adaptive(
            data=data,
            k=self.k,
            distance_fn=self.distance_fn,
            rho=self.rho,
            delta=self.delta,
            max_iterations=self.max_iterations,
            warmup_iterations=self.warmup_iterations,
            sketch_lg_max_k=self.sketch_lg_max_k,
            sample_rate=self.sample_rate,
            seed=self.seed,
            verbose=self.verbose,
        )
        return self

    def get_neighbors(self) -> np.ndarray:
        if self.graph_ is None:
            raise ValueError("Must call fit() first")
        return self.graph_.to_array()

    def get_graph(self) -> KNNGraph:
        return self.graph_
