"""Tests for NN-Descent algorithms and core data structures."""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generator import generate_uniform
from src.distance_metrics import euclidean_distance, cosine_distance, manhattan_distance
from src.knn_graph import KNNGraph, Neighbor
from src.nn_descent_naive import NaiveNNDescent
from src.nn_descent_optimized import OptimizedNNDescent
from src.utils import brute_force_knn, compute_recall


# ── Distance metric tests ────────────────────────────────────────────

class TestDistanceMetrics:

    def test_euclidean(self):
        assert euclidean_distance(np.array([0, 0, 0]), np.array([3, 4, 0])) == 5.0

    def test_euclidean_same_point(self):
        x = np.array([1, 2, 3])
        assert euclidean_distance(x, x) == 0.0

    def test_cosine_orthogonal(self):
        assert cosine_distance(np.array([1, 0]), np.array([0, 1])) == 1.0

    def test_cosine_same_direction(self):
        assert abs(cosine_distance(np.array([1, 0]), np.array([2, 0]))) < 1e-6

    def test_manhattan(self):
        assert manhattan_distance(np.array([0, 0]), np.array([3, 4])) == 7.0


# ── KNNGraph tests ───────────────────────────────────────────────────

class TestKNNGraph:

    def test_initialization(self):
        g = KNNGraph(n=10, k=3)
        assert g.n == 10 and g.k == 3

    def test_random_init_no_self_loops(self):
        g = KNNGraph(n=10, k=3)
        g.initialize_random(exclude_self=True)
        for i in range(10):
            assert i not in g.get_neighbor_indices(i)

    def test_try_update(self):
        g = KNNGraph(n=5, k=2)
        assert g.try_update(0, 1, 1.0) is True
        assert g.try_update(0, 2, 0.5) is True
        assert g.try_update(0, 3, 0.3) is True   # replaces farthest
        assert 3 in g.get_neighbor_indices(0)
        assert g.try_update(0, 4, 2.0) is False   # too far

    def test_reverse_neighbors(self):
        g = KNNGraph(n=5, k=2)
        g.try_update(0, 1, 1.0)
        g.try_update(0, 2, 0.5)
        g.try_update(1, 0, 0.8)
        rev = g.get_reverse_neighbors()
        assert 0 in rev[1]
        assert 0 in rev[2]
        assert 1 in rev[0]


# ── Naive NN-Descent tests ───────────────────────────────────────────

class TestNaiveNNDescent:

    @pytest.fixture
    def small_data(self):
        np.random.seed(42)
        return np.random.rand(50, 10).astype(np.float32)

    def test_produces_valid_graph(self, small_data):
        model = NaiveNNDescent(k=5, seed=42, verbose=False)
        model.fit(small_data)
        nbs = model.get_neighbors()
        assert nbs.shape == (50, 5)
        for i in range(50):
            assert i not in nbs[i]

    def test_recall(self, small_data):
        model = NaiveNNDescent(k=5, seed=42, verbose=False)
        model.fit(small_data)
        gt = brute_force_knn(small_data, 5, euclidean_distance)
        assert compute_recall(model.get_neighbors(), gt) > 0.7


# ── Optimized NN-Descent tests ───────────────────────────────────────

class TestOptimizedNNDescent:

    @pytest.fixture
    def small_data(self):
        np.random.seed(42)
        return np.random.rand(50, 10).astype(np.float32)

    def test_produces_valid_graph(self, small_data):
        model = OptimizedNNDescent(k=5, seed=42, verbose=False)
        model.fit(small_data)
        nbs = model.get_neighbors()
        assert nbs.shape == (50, 5)
        for i in range(50):
            assert i not in nbs[i]

    def test_recall(self, small_data):
        model = OptimizedNNDescent(k=5, rho=0.5, delta=0.001,
                                    seed=42, verbose=False)
        model.fit(small_data)
        gt = brute_force_knn(small_data, 5, euclidean_distance)
        assert compute_recall(model.get_neighbors(), gt) > 0.7

    def test_early_termination(self, small_data):
        model = OptimizedNNDescent(k=5, delta=0.5, max_iterations=100,
                                    seed=42, verbose=False)
        model.fit(small_data)
        assert model.get_neighbors() is not None


# ── Cross-algorithm comparison ────────────────────────────────────────

class TestComparison:

    def test_both_on_same_data(self):
        np.random.seed(42)
        data = np.random.rand(100, 20).astype(np.float32)
        gt = brute_force_knn(data, 10, euclidean_distance)

        naive = NaiveNNDescent(k=10, seed=42, verbose=False)
        optim = OptimizedNNDescent(k=10, seed=42, verbose=False)
        naive.fit(data)
        optim.fit(data)

        assert compute_recall(naive.get_neighbors(), gt) > 0.7
        assert compute_recall(optim.get_neighbors(), gt) > 0.7


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
