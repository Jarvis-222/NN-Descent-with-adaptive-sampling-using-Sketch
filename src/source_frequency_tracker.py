"""
Source Frequency Tracker using Space-Saving Sketch.

Implements per-vertex frequency tracking to identify productive source
neighbors in the NN-Descent algorithm. Uses the datasketches library's 
frequent_strings_sketch for Space-Saving sketch operations.

The key idea: for each point v, we track which of v's neighbors (sources)
most frequently lead to successful graph updates when their neighborhoods
are explored. This information enables adaptive candidate sampling -
candidates from historically productive sources receive higher priority.

"""

from typing import Dict, List, Optional
import numpy as np

try:
    from datasketches import frequent_strings_sketch
    HAS_DATASKETCHES = True
except ImportError:
    HAS_DATASKETCHES = False
    import warnings
    warnings.warn(
        "datasketches library not found — using dictionary fallback. "
        "Install with: pip install datasketches"
    )


class SourceFrequencyTracker:
    """
    Tracks which sources are most productive for each point.

    We use a Space-Saving sketch to estimate the frequency of successful
    updates coming from each neighbor. This helps us prioritize good sources.
    """

    def __init__(self, n_points: int, lg_max_k: int = 6):
        self.n_points = n_points
        self.lg_max_k = lg_max_k
        self._total_updates = 0

        if HAS_DATASKETCHES:
            self._sketches = [
                frequent_strings_sketch(lg_max_k) for _ in range(n_points)
            ]
        else:
            self._counts: List[Dict[str, int]] = [{} for _ in range(n_points)]
            self._max_items = 2 ** lg_max_k

    def record_source_success(self, vertex: int, source: int, weight: int = 1):
        """
        Record that exploring source's neighborhood led to a successful
        update for vertex's k-NN list.

        Args:
            vertex: Point whose neighbor list was improved.
            source: Neighbor whose exploration led to the improvement.
            weight: Number of successes to record.
        """
        key = str(source)

        if HAS_DATASKETCHES:
            for _ in range(weight):
                self._sketches[vertex].update(key)
        else:
            counts = self._counts[vertex]
            counts[key] = counts.get(key, 0) + weight
            if len(counts) > self._max_items:
                self._prune_fallback(vertex)

        self._total_updates += weight

    def _prune_fallback(self, vertex: int):
        """Keep only the top-frequency items in the fallback dictionary."""
        counts = self._counts[vertex]
        if len(counts) <= self._max_items // 2:
            return
        sorted_items = sorted(counts.items(), key=lambda x: -x[1])
        self._counts[vertex] = dict(sorted_items[:self._max_items // 2])

    def get_source_frequency(self, vertex: int, source: int) -> int:
        """
        Get estimated success frequency of a source for a given vertex.

        Args:
            vertex: The point to query.
            source: The source neighbor to check.

        Returns:
            Estimated frequency count (may be approximate with sketches).
        """
        key = str(source)
        if HAS_DATASKETCHES:
            return int(self._sketches[vertex].get_estimate(key))
        else:
            return self._counts[vertex].get(key, 0)

    def get_candidate_weights(
        self,
        vertex: int,
        candidates: List[int],
        candidate_sources: Dict[int, List[int]],
        base_weight: float = 1.0,
        source_boost: float = 0.5,
    ) -> np.ndarray:
        """
        Compute sampling weights for candidates based on their sources'
        historical success rates.

        Weight formula: base_weight + sum(freq(source) * source_boost)
        for each source that discovered this candidate.

        Args:
            vertex: Query point.
            candidates: List of candidate indices.
            candidate_sources: Mapping of candidate → list of sources.
            base_weight: Minimum weight for every candidate.
            source_boost: Multiplier per source success count.

        Returns:
            Array of sampling weights, one per candidate.
        """
        weights = np.empty(len(candidates), dtype=np.float64)

        for i, c in enumerate(candidates):
            w = base_weight
            for src in candidate_sources.get(c, []):
                w += self.get_source_frequency(vertex, src) * source_boost
            weights[i] = w

        return weights

    def sample_candidates(
        self,
        vertex: int,
        candidates: List[int],
        candidate_sources: Dict[int, List[int]],
        sample_size: int,
        rng: np.random.Generator = None,
        base_weight: float = 1.0,
        source_boost: float = 0.5,
    ) -> List[int]:
        """
        Sample candidates with probability proportional to source success.

        Candidates from historically productive sources are more likely
        to be selected, while all candidates maintain a nonzero probability
        to preserve exploration.

        Args:
            vertex: Query point.
            candidates: Full list of candidate indices.
            candidate_sources: Mapping of candidate → source neighbors.
            sample_size: Number of candidates to select.
            rng: NumPy random generator.
            base_weight: Minimum sampling weight.
            source_boost: Weight multiplier per source success.

        Returns:
            List of selected candidate indices.
        """
        if rng is None:
            rng = np.random.default_rng()

        if len(candidates) <= sample_size:
            return candidates

        weights = self.get_candidate_weights(
            vertex, candidates, candidate_sources,
            base_weight=base_weight, source_boost=source_boost,
        )

        total = weights.sum()
        probs = weights / total if total > 0 else np.ones(len(candidates)) / len(candidates)

        indices = rng.choice(len(candidates), size=sample_size, replace=False, p=probs)
        return [candidates[i] for i in indices]

    def sample_neighbors(
        self,
        vertex: int,
        neighbors: List[int],
        sample_size: int,
        rng: np.random.Generator = None,
        base_weight: float = 1.0,
        source_boost: float = 0.5,
    ) -> List[int]:
        """
        Sketch-weighted ρ-sampling of neighbor lists.

        Selects which neighbors to explore based on their historical
        success as sources. Neighbors that led to more graph updates
        are more likely to be kept.

        Args:
            vertex: The point whose neighbor list is being sampled.
            neighbors: Full list of neighbor indices.
            sample_size: Number of neighbors to keep (ρ × k).
            rng: NumPy random generator.
            base_weight: Minimum weight for every neighbor.
            source_boost: Weight multiplier per success count.

        Returns:
            List of selected neighbor indices.
        """
        if rng is None:
            rng = np.random.default_rng()

        if len(neighbors) <= sample_size:
            return neighbors

        weights = np.empty(len(neighbors), dtype=np.float64)
        for i, nb in enumerate(neighbors):
            freq = self.get_source_frequency(vertex, nb)
            weights[i] = base_weight + freq * source_boost

        total = weights.sum()
        probs = weights / total if total > 0 else np.ones(len(neighbors)) / len(neighbors)

        indices = rng.choice(len(neighbors), size=sample_size, replace=False, p=probs)
        return [neighbors[i] for i in indices]

    @property
    def total_updates(self) -> int:
        """Total source successes recorded across all vertices."""
        return self._total_updates

    def reset(self):
        """Clear all tracked frequencies."""
        if HAS_DATASKETCHES:
            self._sketches = [
                frequent_strings_sketch(self.lg_max_k) for _ in range(self.n_points)
            ]
        else:
            self._counts = [{} for _ in range(self.n_points)]
        self._total_updates = 0