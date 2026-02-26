"""
Distance metrics for k-NN graph construction.

Provides pluggable distance functions for nearest neighbor search.
Each function takes two NumPy vectors and returns a scalar distance.

"""

import numpy as np
from typing import Callable


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """L2 or Euclidain distance... """
    return float(np.sqrt(np.sum((x - y) ** 2)))


def cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Cosine distance... """
    dot = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    if norm_x == 0 or norm_y == 0:
        return 1.0
    return 1.0 - (dot / (norm_x * norm_y))


def manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
    """L1 or Manhattan distance..."""
    return float(np.sum(np.abs(x - y)))


DISTANCE_FUNCTIONS = {
    'euclidean': euclidean_distance,
    'cosine': cosine_distance,
    'manhattan': manhattan_distance,
}


def get_distance_function(name: str) -> Callable:
    """
    Args:
    name :'euclidean', 'cosine', 'manhattan'.

    Returns:
        The corresponding distance function.

    Raises:
        ValueError: If the name is not recognized.
    """
    if name not in DISTANCE_FUNCTIONS:
        raise ValueError(
            f"Unknown distance function: {name}. "
            f"Available: {list(DISTANCE_FUNCTIONS.keys())}"
        )
    return DISTANCE_FUNCTIONS[name]
