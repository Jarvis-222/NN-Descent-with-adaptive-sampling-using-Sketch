"""
Utility functions for NN-Descent experiments.

Provides brute-force k-NN computation (for ground truth), recall
evaluation, and benchmarking helpers.
"""

import numpy as np
from typing import Callable
import time


def brute_force_knn(data: np.ndarray, k: int, distance_fn: Callable) -> np.ndarray:
    """
    Compute exact k-NN graph using brute force. Used as ground truth.

    Args:
        data: Dataset of shape (n, dim).
        k: Number of nearest neighbors.
        distance_fn: Pairwise distance function.

    Returns:
        Array of shape (n, k) with ground-truth neighbor indices.
    """
    n = len(data)
    neighbors = np.zeros((n, k), dtype=np.int32)

    for i in range(n):
        distances = np.array([
            distance_fn(data[i], data[j]) if i != j else float('inf')
            for j in range(n)
        ])
        neighbors[i] = np.argsort(distances)[:k]

    return neighbors


def compute_recall(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Compute recall between predicted and ground-truth k-NN graphs.

    Recall = (number of correct neighbors) / (total neighbors).

    Args:
        predicted: Predicted k-NN indices, shape (n, k).
        ground_truth: True k-NN indices, shape (n, k).

    Returns:
        Recall score in [0, 1].
    """
    n = len(predicted)
    correct = 0
    total = 0

    for i in range(n):
        correct += len(set(predicted[i]) & set(ground_truth[i]))
        total += len(ground_truth[i])

    return correct / total if total > 0 else 0.0


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str = ""):
        self.name = name
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self._start
        if self.name:
            print(f"{self.name}: {self.elapsed:.4f}s")
