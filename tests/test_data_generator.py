"""Tests for data generators."""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generator import generate_uniform, generate_gaussian, generate_clustered


class TestGenerateUniform:

    def test_correct_shape(self):
        data = generate_uniform(100, 50, seed=42)
        assert data.shape == (100, 50)

    def test_correct_range(self):
        data = generate_uniform(1000, 10, low=0, high=1, seed=42)
        assert np.all(data >= 0)
        assert np.all(data <= 1)

    def test_reproducibility(self):
        d1 = generate_uniform(50, 10, seed=42)
        d2 = generate_uniform(50, 10, seed=42)
        np.testing.assert_array_equal(d1, d2)


class TestGenerateGaussian:

    def test_correct_shape(self):
        data = generate_gaussian(100, 50, seed=42)
        assert data.shape == (100, 50)

    def test_approximate_statistics(self):
        data = generate_gaussian(10000, 10, mean=5, std=2, seed=42)
        assert abs(np.mean(data) - 5) < 0.1
        assert abs(np.std(data) - 2) < 0.1


class TestGenerateClustered:

    def test_correct_shape(self):
        data = generate_clustered(100, 50, n_clusters=5, seed=42)
        assert data.shape == (100, 50)

    def test_clusters_are_tight(self):
        data = generate_clustered(300, 2, n_clusters=3, cluster_std=0.01, seed=42)
        for start in [0, 100, 200]:
            cluster = data[start:start + 100]
            center = np.mean(cluster, axis=0)
            distances = np.sqrt(np.sum((cluster - center) ** 2, axis=1))
            assert np.max(distances) < 0.1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
