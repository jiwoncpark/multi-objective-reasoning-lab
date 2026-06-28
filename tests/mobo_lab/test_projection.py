"""Tests for ``mobo_lab/projection.py`` (continuous proposal -> pool row)."""

from __future__ import annotations

import pytest
import torch

from mobo_lab import projection

# A tiny, hand-laid 2-D pool so the geometry is readable. The four corners of the
# unit square plus its centre.
SQUARE = torch.tensor(
    [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.5, 0.5]], dtype=torch.double
)


def _min_pairwise(points: torch.Tensor) -> float:
    """Smallest Euclidean distance between any two distinct rows."""
    d = torch.cdist(points, points)
    d.fill_diagonal_(float("inf"))
    return float(d.min())


def test_nearest_snaps_to_closest_row():
    # [0.1, 0.1] is closest to the origin corner (row 0).
    assert projection.nearest([[0.1, 0.1]], SQUARE, set()) == [0]


def test_nearest_falls_back_to_next_closest_when_forbidden():
    # With row 0 forbidden, the next-closest row to [0.1, 0.1] is the centre (4).
    assert projection.nearest([[0.1, 0.1]], SQUARE, {0}) == [4]


def test_identical_candidates_map_to_distinct_rows():
    # Two identical proposals must not collapse onto the same ID.
    ids = projection.nearest([[0.1, 0.1], [0.1, 0.1]], SQUARE, set())
    assert ids == [0, 4]
    assert len(set(ids)) == 2


def test_identity_lookup_for_exact_pool_rows():
    # Feeding the pool rows themselves back in returns their indices in order
    # (each is its own zero-distance nearest neighbour) -- the property that makes
    # the discrete golden path's projection exact.
    assert projection.nearest(SQUARE, SQUARE, set()) == [0, 1, 2, 3, 4]


def test_forbidden_set_not_mutated():
    forbidden = {0}
    projection.nearest([[0.1, 0.1], [0.9, 0.9]], SQUARE, forbidden)
    assert forbidden == {0}  # caller's set is left untouched


def test_nearest_raises_when_pool_exhausted():
    with pytest.raises(ValueError, match="available pool rows"):
        projection.nearest([[0.1, 0.1]], SQUARE, {0, 1, 2, 3, 4})


def test_diverse_nearest_spreads_a_clustered_batch():
    # A pool with two near-origin rows (0, 1) and one far row (2). Two proposals
    # both sit near the origin cluster.
    pool = torch.tensor([[0.0, 0.0], [0.2, 0.0], [0.0, 1.0]], dtype=torch.double)
    cands = [[0.0, 0.0], [0.15, 0.5]]

    plain = projection.nearest(cands, pool, set())
    diverse = projection.diverse_nearest(cands, pool, set())

    # Plain nearest grabs the two crowded origin rows; diverse trades the second
    # pick for the far row, so the chosen set is strictly more spread out.
    assert plain == [0, 1]
    assert diverse == [0, 2]
    assert _min_pairwise(pool[diverse]) >= _min_pairwise(pool[plain])


def test_diverse_nearest_matches_nearest_for_single_candidate():
    # With nothing yet chosen there is no diversity term, so the first pick is
    # identical to plain nearest.
    cand = [[0.1, 0.1]]
    assert projection.diverse_nearest(cand, SQUARE, set()) == projection.nearest(
        cand, SQUARE, set()
    )


def test_methods_dispatch_table():
    assert set(projection.METHODS) == {"nearest", "diverse_nearest"}


def test_single_1d_candidate_accepted():
    assert projection.nearest([0.1, 0.1], SQUARE, set()) == [0]


def test_wrong_dimension_raises():
    with pytest.raises(ValueError, match="shape"):
        projection.nearest([[0.1, 0.1, 0.1]], SQUARE, set())
