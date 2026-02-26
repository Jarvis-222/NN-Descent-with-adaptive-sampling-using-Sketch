"""
k-NN Graph data structure for NN-Descent.

Stores approximate nearest neighbors for each point with distances
and new/old flags used by the incremental search optimization.
"""

import numpy as np
from typing import List, Set, Callable
from dataclasses import dataclass


@dataclass
class Neighbor:
    """A neighbor entry: stores index, distance, and a new/old flag."""
    index: int
    distance: float
    is_new: bool = True

    def __lt__(self, other):
        return self.distance < other.distance

    def __eq__(self, other):
        return isinstance(other, Neighbor) and self.index == other.index

    def __hash__(self):
        return hash(self.index)



class KNNGraph:
    """
    A simple k-NN graph using sorted lists of neighbors.
    """

    def __init__(self, n: int, k: int):
        self.n = n
        self.k = k
        self.neighbors: List[List[Neighbor]] = [[] for _ in range(n)]

    def initialize_random(self, exclude_self: bool = True) -> None:
        """Randomly initialize the graph."""
        for i in range(self.n):
            candidates = list(range(self.n))
            if exclude_self:
                candidates.remove(i)
            chosen = np.random.choice(
                candidates, size=min(self.k, len(candidates)), replace=False
            )
            for j in chosen:
                self.neighbors[i].append(
                    Neighbor(index=j, distance=float('inf'), is_new=True)
                )

    def update_distances(self, data: np.ndarray, distance_fn: Callable) -> None:
        """Calculate real distances for the random neighbors."""
        for i in range(self.n):
            for nb in self.neighbors[i]:
                nb.distance = distance_fn(data[i], data[nb.index])
            self.neighbors[i].sort(key=lambda x: x.distance)

    def try_update(self, point: int, candidate: int, dist: float):
        """
        Try to add a candidate neighbor.

        Returns:
            None  – no update (duplicate or worse than all current).
            -1    – added successfully (list had space, nothing evicted).
            int   – index of the evicted neighbor that was replaced.
        """
        if point == candidate:
            return None

        nbs = self.neighbors[point]

        # Check if already in list
        for nb in nbs:
            if nb.index == candidate:
                return None

        # If we have space, just add it
        if len(nbs) < self.k:
            nbs.append(Neighbor(index=candidate, distance=dist, is_new=True))
            nbs.sort(key=lambda x: x.distance)
            return -1

        # If list is full, replace the worst one if this is better
        if dist < nbs[-1].distance:
            evicted = nbs[-1].index
            nbs[-1] = Neighbor(index=candidate, distance=dist, is_new=True)
            nbs.sort(key=lambda x: x.distance)
            return evicted

        return None

    def get_neighbors(self, point: int) -> List[Neighbor]:
        return self.neighbors[point]

    def get_neighbor_indices(self, point: int) -> List[int]:
        return [nb.index for nb in self.neighbors[point]]

    def get_new_neighbors(self, point: int) -> List[Neighbor]:
        return [nb for nb in self.neighbors[point] if nb.is_new]

    def get_old_neighbors(self, point: int) -> List[Neighbor]:
        return [nb for nb in self.neighbors[point] if not nb.is_new]

    def get_reverse_neighbors(self) -> List[Set[int]]:
        """Find which points have me as a neighbor."""
        reverse = [set() for _ in range(self.n)]
        for i in range(self.n):
            for nb in self.neighbors[i]:
                reverse[nb.index].add(i)
        return reverse

    def mark_all_old(self) -> None:
        for i in range(self.n):
            for nb in self.neighbors[i]:
                nb.is_new = False

    def to_array(self) -> np.ndarray:
        """Return the graph as a numpy array of indices."""
        result = np.zeros((self.n, self.k), dtype=np.int32)
        for i in range(self.n):
            for j, nb in enumerate(self.neighbors[i][:self.k]):
                result[i, j] = nb.index
        return result

    def compute_recall(self, ground_truth: np.ndarray) -> float:
        """Calculate recall compared to ground truth."""
        correct = total = 0
        for i in range(self.n):
            pred = set(self.get_neighbor_indices(i)[:len(ground_truth[i])])
            true = set(ground_truth[i])
            correct += len(pred & true)
            total += len(true)
        return correct / total if total > 0 else 0.0

