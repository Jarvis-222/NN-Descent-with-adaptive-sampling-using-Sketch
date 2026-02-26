"""
Optimized NN-Descent with Local Join and Incremental Search.

Implements the full set of optimizations from the NN-Descent paper:
  1. Local Join
  2. Incremental Search
  3. Reverse Neighbors
  4. Sampling
  5. Early Termination

Reference:
    Dong et al., "Efficient K-Nearest Neighbor Graph Construction for
    Generic Similarity Measures", WWW 2011.
"""

import numpy as np
from typing import Callable, Optional
from .knn_graph import KNNGraph


def nn_descent_optimized(
    data: np.ndarray,
    k: int,
    distance_fn: Callable,
    rho: float = 0.5,
    delta: float = 0.001,
    max_iterations: int = 20,
    sample_rate: float = 1.0,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> KNNGraph:
    """
    Optimized NN-Descent algorithm.

    Collects candidates from neighbors of neighbors.
    Uses 'Local Join', 'Incremental Search', and 'Reverse Neighbors'
    to speed up convergence.
    """
    rng = np.random.default_rng(seed)
    if seed is not None:
        np.random.seed(seed)

    n = len(data)
    sample_size = max(1, int(rho * k))
    termination_threshold = delta * n * k

    if verbose:
        print(f"Optimized NN-Descent: n={n}, k={k}, ρ={rho}, δ={delta}")
        print(f"  Sample size: {sample_size}, "
              f"Termination threshold: {termination_threshold:.1f}")
        if sample_rate < 1.0:
            print(f"  Candidate sample rate: {sample_rate:.0%}")

    # *********Initialization*********
    graph = KNNGraph(n, k)
    graph.initialize_random(exclude_self=True)
    graph.update_distances(data, distance_fn)

    distance_computations = 0

    # **********Main loop **********
    for iteration in range(max_iterations):
        updates = 0

        #Build reverse neighbour look-up
        reverse = graph.get_reverse_neighbors()

        # Categorise each point's neighbours into NEW and OLD lists,
        # incorporating reverse neighbours, and sample down.
        new_lists, old_lists = [], []

        for v in range(n):
            new_nbs = [nb.index for nb in graph.get_new_neighbors(v)]
            old_nbs = [nb.index for nb in graph.get_old_neighbors(v)]

            # Merge reverse neighbors into the appropriate list
            for u in reverse[v]:
                is_new_reverse = any(
                    nb.index == v for nb in graph.get_new_neighbors(u)
                )
                target = new_nbs if is_new_reverse else old_nbs
                if u not in target:
                    target.append(u)

            # Sample down if too large
            if len(new_nbs) > sample_size:
                new_nbs = list(np.random.choice(new_nbs, sample_size, replace=False))
            if len(old_nbs) > sample_size:
                old_nbs = list(np.random.choice(old_nbs, sample_size, replace=False))

            new_lists.append(new_nbs)
            old_lists.append(old_nbs)

        # Snapshot new-neighbor sets *before* marking everything old
        new_sets = [set(new_lists[i]) for i in range(n)]
        graph.mark_all_old()

        # ── Local Join: candidate discovery ───────────────────────────
        for v in range(n):
            current_nbs = set(graph.get_neighbor_indices(v))
            candidates = set()

            # NEW neighbours → explore ALL their neighbours
            for u in new_lists[v]:
                for nb in graph.get_neighbors(u):
                    if nb.index != v and nb.index not in current_nbs:
                        candidates.add(nb.index)

            # OLD neighbours → explore only their NEW neighbours
            for u in old_lists[v]:
                for w in new_sets[u]:
                    if w != v and w not in current_nbs:
                        candidates.add(w)

            # Evaluate candidates
            for c in candidates:
                dist = distance_fn(data[v], data[c])
                distance_computations += 1
                if graph.try_update(v, c, dist) is not None:
                    updates += 1
                if graph.try_update(c, v, dist) is not None:
                    updates += 1

        if verbose:
            print(f"  Iteration {iteration + 1}: {updates} updates")

        if updates < termination_threshold:
            if verbose:
                print(f"  Early termination at iteration {iteration + 1} "
                      f"(updates {updates} < {termination_threshold:.0f})")
            break

    if verbose:
        print(f"  Total distance computations: {distance_computations}")

    return graph


class OptimizedNNDescent:
    """ Wrapper for optimized NN-Descent."""

    def __init__(
        self,
        k: int = 10,
        distance_fn: Callable = None,
        rho: float = 0.5,
        delta: float = 0.001,
        max_iterations: int = 20,
        sample_rate: float = 1.0,
        seed: Optional[int] = None,
        verbose: bool = True,
    ):
        self.k = k
        self.distance_fn = distance_fn
        self.rho = rho
        self.delta = delta
        self.max_iterations = max_iterations
        self.sample_rate = sample_rate
        self.seed = seed
        self.verbose = verbose
        self.graph_: Optional[KNNGraph] = None

    def fit(self, data: np.ndarray) -> "OptimizedNNDescent":
        if self.distance_fn is None:
            from .distance_metrics import euclidean_distance
            self.distance_fn = euclidean_distance

        self.graph_ = nn_descent_optimized(
            data=data,
            k=self.k,
            distance_fn=self.distance_fn,
            rho=self.rho,
            delta=self.delta,
            max_iterations=self.max_iterations,
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
