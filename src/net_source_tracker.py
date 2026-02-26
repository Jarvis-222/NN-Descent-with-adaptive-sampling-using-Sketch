"""
Net Source Tracker with Eviction Penalty.

A dictionary-based variant of SourceFrequencyTracker that supports both
incrementing scores (on successful updates) and decrementing them
(when a placed neighbor is later evicted). This gives a net-benefit
signal rather than a gross-success signal.

Unlike SourceFrequencyTracker which uses a Space-Saving sketch (append
only), this class uses plain dictionaries so that scores can go both up
and down. Scores are clamped at 0.
"""

from typing import Dict, List, Optional
import numpy as np
import math


class NetSourceTracker:
    """
    Tracks net source productivity (successes minus eviction penalties).

    Uses dictionary-based counting so that scores can be both incremented
    and decremented. Scores are clamped at 0.
    """

    def __init__(self, n_points: int, lg_max_k: int = 6):
        self.n_points = n_points
        self.lg_max_k = lg_max_k
        self._max_items = 2 ** lg_max_k
        self._total_updates = 0
        self._counts: List[Dict[str, int]] = [{} for _ in range(n_points)]

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
        counts = self._counts[vertex]
        counts[key] = counts.get(key, 0) + weight
        if len(counts) > self._max_items:
            self._prune(vertex)
        self._total_updates += weight

    def record_source_penalty(self, vertex: int, source: int, weight: int = 1):
        """
        Penalise a source because a neighbor it placed was evicted.

        The score is decremented but clamped to 0.

        Args:
            vertex: Point whose neighbor list changed.
            source: Source that originally placed the evicted neighbor.
            weight: Penalty amount.
        """
        key = str(source)
        counts = self._counts[vertex]
        if key in counts:
            counts[key] = max(0, counts[key] - weight)
            if counts[key] == 0:
                del counts[key]

    def _prune(self, vertex: int):
        """Keep only the top-frequency items."""
        counts = self._counts[vertex]
        if len(counts) <= self._max_items // 2:
            return
        sorted_items = sorted(counts.items(), key=lambda x: -x[1])
        self._counts[vertex] = dict(sorted_items[:self._max_items // 2])

    def get_source_frequency(self, vertex: int, source: int) -> int:
        """
        Get net success frequency of a source for a given vertex.

        Args:
            vertex: The point to query.
            source: The source neighbor to check.

        Returns:
            Frequency count (net of penalties), always >= 0.
        """
        key = str(source)
        return self._counts[vertex].get(key, 0)

    def get_candidate_weights(
        self,
        vertex: int,
        candidates: List[int],
        candidate_sources: Dict[int, List[int]],
        base_weight: float = 1.0,
        source_boost: float = 0.5,
        in_degrees: Optional[Dict[int, int]] = None,
        k: int = 10,
    ) -> np.ndarray:
        """
        Compute sampling weights for candidates based on their sources'
        net success, with optional hub penalty.

        Weight formula per candidate c:
          base_weight + sum( freq(src) * source_boost / hub_penalty(src) )

        Hub penalty for source s:
          1 + log(in_degree(s) / k)   if in_degrees provided and in_degree > k
          1.0                          otherwise (no penalty)

        This auto-detects hubs: sources appearing in many NN lists
        (in_degree >> k) get dampened, while sources with normal
        in-degree (~k) pass through unaffected.
        """
        weights = np.empty(len(candidates), dtype=np.float64)

        for i, c in enumerate(candidates):
            w = base_weight
            for src in candidate_sources.get(c, []):
                freq = self.get_source_frequency(vertex, src)
                if in_degrees is not None and freq > 0:
                    deg = in_degrees.get(src, k)
                    ratio = deg / k
                    # Squared penalty: hubs get heavily dampened
                    hub_penalty = ratio * ratio if ratio > 1.0 else 1.0
                    w += (freq / hub_penalty) * source_boost
                else:
                    w += freq * source_boost
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
        in_degrees: Optional[Dict[int, int]] = None,
        k: int = 10,
    ) -> List[int]:
        """
        Sample candidates weighted by net source success with hub penalty.

        Candidates from productive sources are boosted, but sources that
        appear in too many NN lists (hubs) have their boost dampened.
        All candidates keep a nonzero base probability for exploration.
        """
        if rng is None:
            rng = np.random.default_rng()

        if len(candidates) <= sample_size:
            return candidates

        weights = self.get_candidate_weights(
            vertex, candidates, candidate_sources,
            base_weight=base_weight, source_boost=source_boost,
            in_degrees=in_degrees, k=k,
        )

        total = weights.sum()
        probs = weights / total if total > 0 else np.ones(len(candidates)) / len(candidates)

        indices = rng.choice(len(candidates), size=sample_size, replace=False, p=probs)
        return [candidates[i] for i in indices]

    @property
    def total_updates(self) -> int:
        """Total source successes recorded across all vertices."""
        return self._total_updates

    def reset(self):
        """Clear all tracked frequencies."""
        self._counts = [{} for _ in range(self.n_points)]
        self._total_updates = 0
