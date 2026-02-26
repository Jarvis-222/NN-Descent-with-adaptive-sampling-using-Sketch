"""Simple data generators for k-NN experiments."""

import numpy as np
import struct
import os
from typing import Optional


def generate_uniform(n: int, dim: int, low=0.0, high=1.0, seed=None):
    """Generate uniform random data in [low, high)."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(low, high, (n, dim)).astype(np.float32)


def generate_gaussian(n: int, dim: int, mean=0.0, std=1.0, seed=None):
    """Generate Gaussian distributed random data."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.normal(mean, std, (n, dim)).astype(np.float32)


def generate_high_hubness(n: int, dim: int = 100, seed=None):
    """
    Generate data that exhibits strong hubness.

    Uses isotropic Gaussian in high dimensions (default d=100).
    In high-d, pairwise distances concentrate around the same value,
    so points near the centroid dominate everyone's NN lists (hubs)
    while peripheral points rarely appear (anti-hubs).

    This is the setting from Bratic et al. where standard NN-Descent
    drops to ~9% recall at k=5.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)



def load_sift(path: str, n: Optional[int] = None):
    """Load SIFT dataset from .fvecs file. Each vector: [dim (int32)] [float32 × dim]."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"SIFT file not found: {path}")

    vectors = []
    with open(path, 'rb') as f:
        while True:
            buf = f.read(4)
            if not buf:
                break
            dim = struct.unpack('i', buf)[0]
            vec = f.read(dim * 4)
            if len(vec) != dim * 4:
                break
            vectors.append(struct.unpack('f' * dim, vec))

    data = np.array(vectors, dtype=np.float32)
    if n is not None and n < len(data):
        data = data[:n]
    return data
