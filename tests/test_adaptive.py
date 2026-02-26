"""Tests for Source Frequency Tracker and Adaptive NN-Descent."""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.source_frequency_tracker import SourceFrequencyTracker
from src.nn_descent_adaptive import AdaptiveNNDescent
from src.distance_metrics import euclidean_distance
from src.utils import brute_force_knn, compute_recall


# ── SourceFrequencyTracker tests ─────────────────────────────────────

class TestSourceFrequencyTracker:

    def test_basic_recording(self):
        t = SourceFrequencyTracker(n_points=100, lg_max_k=6)
        t.record_source_success(0, source=1)
        t.record_source_success(0, source=1)
        t.record_source_success(0, source=2)
        assert t.get_source_frequency(0, 1) == 2
        assert t.get_source_frequency(0, 2) == 1
        assert t.get_source_frequency(0, 3) == 0

    def test_per_vertex_isolation(self):
        t = SourceFrequencyTracker(n_points=100, lg_max_k=6)
        t.record_source_success(0, source=1, weight=10)
        t.record_source_success(1, source=1, weight=3)
        assert t.get_source_frequency(0, 1) == 10
        assert t.get_source_frequency(1, 1) == 3

    def test_total_updates(self):
        t = SourceFrequencyTracker(n_points=100)
        t.record_source_success(0, 1, weight=5)
        t.record_source_success(1, 2, weight=3)
        assert t.total_updates == 8

    def test_candidate_weights(self):
        t = SourceFrequencyTracker(n_points=100, lg_max_k=6)
        t.record_source_success(0, source=1, weight=10)
        t.record_source_success(0, source=2, weight=1)
        sources = {10: [1], 20: [2], 30: []}
        w = t.get_candidate_weights(0, [10, 20, 30], sources)
        assert w[0] > w[1]   # candidate from productive source
        assert w[2] == 1.0   # base weight for unknown source

    def test_sample_candidates(self):
        t = SourceFrequencyTracker(n_points=100, lg_max_k=6)
        rng = np.random.default_rng(42)
        t.record_source_success(0, source=1, weight=100)

        candidates = [10, 20, 30, 40, 50]
        sources = {10: [1], 20: [], 30: [], 40: [], 50: []}
        counts = {c: 0 for c in candidates}
        for _ in range(100):
            for s in t.sample_candidates(0, candidates, sources, 2, rng):
                counts[s] += 1
        assert counts[10] > counts[50]

    def test_reset(self):
        t = SourceFrequencyTracker(n_points=100)
        t.record_source_success(0, 1, weight=10)
        assert t.total_updates == 10
        t.reset()
        assert t.total_updates == 0
        assert t.get_source_frequency(0, 1) == 0


# ── Adaptive NN-Descent tests ───────────────────────────────────────

class TestAdaptiveNNDescent:

    @pytest.fixture
    def small_data(self):
        np.random.seed(42)
        return np.random.rand(50, 10).astype(np.float32)

    def test_produces_valid_graph(self, small_data):
        model = AdaptiveNNDescent(k=5, seed=42, verbose=False)
        model.fit(small_data)
        nbs = model.get_neighbors()
        assert nbs.shape == (50, 5)
        for i in range(50):
            assert i not in nbs[i]

    def test_recall(self, small_data):
        model = AdaptiveNNDescent(k=5, seed=42, verbose=False)
        model.fit(small_data)
        gt = brute_force_knn(small_data, 5, euclidean_distance)
        assert compute_recall(model.get_neighbors(), gt) > 0.7

    def test_warmup_iterations(self, small_data):
        m1 = AdaptiveNNDescent(k=5, warmup_iterations=0, seed=42, verbose=False)
        m2 = AdaptiveNNDescent(k=5, warmup_iterations=5, seed=42, verbose=False)
        m1.fit(small_data)
        m2.fit(small_data)
        assert m1.get_neighbors().shape == (50, 5)
        assert m2.get_neighbors().shape == (50, 5)

    def test_sketch_size(self, small_data):
        m1 = AdaptiveNNDescent(k=5, sketch_lg_max_k=4, seed=42, verbose=False)
        m2 = AdaptiveNNDescent(k=5, sketch_lg_max_k=8, seed=42, verbose=False)
        m1.fit(small_data)
        m2.fit(small_data)
        assert m1.get_neighbors().shape == (50, 5)
        assert m2.get_neighbors().shape == (50, 5)

    def test_sample_rate(self, small_data):
        m1 = AdaptiveNNDescent(k=5, sample_rate=1.0, seed=42, verbose=False)
        m2 = AdaptiveNNDescent(k=5, sample_rate=0.5, seed=42, verbose=False)
        m1.fit(small_data)
        m2.fit(small_data)
        assert m1.get_neighbors().shape == (50, 5)
        assert m2.get_neighbors().shape == (50, 5)


# ── Cross-algorithm comparison (all three) ───────────────────────────

class TestAllAlgorithms:

    def test_all_on_same_data(self):
        np.random.seed(42)
        data = np.random.rand(100, 20).astype(np.float32)
        gt = brute_force_knn(data, 10, euclidean_distance)

        from src.nn_descent_naive import NaiveNNDescent
        from src.nn_descent_optimized import OptimizedNNDescent

        for Model in [NaiveNNDescent, OptimizedNNDescent, AdaptiveNNDescent]:
            model = Model(k=10, seed=42, verbose=False)
            model.fit(data)
            assert compute_recall(model.get_neighbors(), gt) > 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
