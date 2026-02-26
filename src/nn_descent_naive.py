"""
Naive NN-Descent — baseline implementation without optimizations.

Iterates through all neighbors of neighbors and updates the graph
if a closer neighbor is found. No local join, no sampling, no
incremental search.

This serves as a correctness baseline for the optimized variants.

Reference:
    Dong et al., "Efficient K-Nearest Neighbor Graph Construction for
    Generic Similarity Measures", WWW 2011, Section 3.
"""

import numpy as np
from typing import Callable, Optional
from .knn_graph import KNNGraph


def nn_descent_naive(
    data: np.ndarray,
    k: int,
    distance_fn: Callable,
    max_iterations: int = 20,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> KNNGraph:
    """
    Standard Naive NN-Descent.
    
    Checks every neighbor-of-neighbor without any optimization.
    Good for checking correctness of other versions.
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(data)

    if verbose:
        print(f"Naive NN-Descent: n={n}, k={k}")

    # Step 1: Random initialization
    graph = KNNGraph(n, k)
    graph.initialize_random(exclude_self=True)
    graph.update_distances(data, distance_fn)

    distance_computations = 0

    # Step 2: Iterate
    for iteration in range(max_iterations):
        updates = 0

        for v in range(n):
            # For each neighbor u of v
            for nb_u in graph.get_neighbors(v):
                u = nb_u.index
                # Examine u's neighbors as candidates for v
                for nb_w in graph.get_neighbors(u):
                    w = nb_w.index
                    if w == v:
                        continue

                    dist = distance_fn(data[v], data[w])
                    distance_computations += 1

                    if graph.try_update(v, w, dist) is not None:
                        updates += 1
                    if graph.try_update(w, v, dist) is not None:
                        updates += 1

        if verbose:
            print(f"  Iteration {iteration + 1}: {updates} updates")

        if updates == 0:
            if verbose:
                print(f"  Converged at iteration {iteration + 1}")
            break

    if verbose:
        print(f"  Total distance computations: {distance_computations}")

    return graph


class NaiveNNDescent:
    """Scikit-learn-style wrapper for naive NN-Descent."""

    def __init__(
        self,
        k: int = 10,
        distance_fn: Callable = None,
        max_iterations: int = 20,
        seed: Optional[int] = None,
        verbose: bool = True,
    ):
        self.k = k
        self.distance_fn = distance_fn
        self.max_iterations = max_iterations
        self.seed = seed
        self.verbose = verbose
        self.graph_: Optional[KNNGraph] = None

    def fit(self, data: np.ndarray) -> "NaiveNNDescent":
        if self.distance_fn is None:
            from .distance_metrics import euclidean_distance
            self.distance_fn = euclidean_distance

        self.graph_ = nn_descent_naive(
            data=data,
            k=self.k,
            distance_fn=self.distance_fn,
            max_iterations=self.max_iterations,
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
