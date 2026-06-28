"""Tests for the pure helpers in ``scripts/visualize_data.py``.

We only test the data logic (the 2D Pareto-front helpers); the plotting
functions themselves write PNGs and are exercised by running the script. The
cases below use tiny hand-checkable point sets.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from visualize_data import (  # noqa: E402
    _pareto_front_2d_max,
    _pareto_staircase,
    nearest_neighbor_distances,
)


def test_pareto_front_simple_maximization() -> None:
    # Both axes higher=better. (2,2) dominates (1,1); (0,3) and (3,0) are the
    # extreme trade-offs; (1,1) is dominated.
    x = np.array([2.0, 1.0, 0.0, 3.0])
    y = np.array([2.0, 1.0, 3.0, 0.0])
    mask = _pareto_front_2d_max(x, y)
    # Non-dominated: (2,2), (0,3), (3,0). Dominated: (1,1).
    assert mask.tolist() == [True, False, True, True]


def test_pareto_front_all_nondominated_on_antidiagonal() -> None:
    # Strict trade-off: as x increases y decreases -> every point is on the front.
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([3.0, 2.0, 1.0, 0.0])
    assert _pareto_front_2d_max(x, y).all()


def test_pareto_front_single_dominator() -> None:
    # One point beats everything on both axes.
    x = np.array([10.0, 1.0, 2.0, 3.0])
    y = np.array([10.0, 1.0, 2.0, 3.0])
    assert _pareto_front_2d_max(x, y).tolist() == [True, False, False, False]


def test_staircase_is_monotonic() -> None:
    # Front points (sorted by x) should yield x non-decreasing and y non-increasing.
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([3.0, 2.0, 1.0, 0.0])
    sx, sy = _pareto_staircase(x, y)
    assert np.all(np.diff(sx) >= 0)
    assert np.all(np.diff(sy) <= 0)
    # Starts at the highest-y point and ends at the highest-x point.
    assert (sx[0], sy[0]) == (0.0, 3.0)
    assert (sx[-1], sy[-1]) == (3.0, 0.0)


def test_staircase_uses_inner_corners() -> None:
    # Between (0.6, 0.95) and (0.9, 0.5) the corner must be the inner (0.6, 0.5),
    # not a phantom outer corner at (0.9, 0.95) above both points.
    x = np.array([0.6, 0.9])
    y = np.array([0.95, 0.5])
    sx, sy = _pareto_staircase(x, y)
    corners = list(zip(sx.tolist(), sy.tolist()))
    assert corners == [(0.6, 0.95), (0.6, 0.5), (0.9, 0.5)]
    assert (0.9, 0.95) not in corners
    # the polyline never rises above the points' bounding box
    assert sy.max() == 0.95 and sx.max() == 0.9


def test_nearest_neighbor_distances_known_points() -> None:
    # NN of (0,0) is (0,3) at 3; of (0,3) is (0,0) at 3; of (4,0) is (0,0) at 4.
    pts = np.array([[0.0, 0.0], [0.0, 3.0], [4.0, 0.0]])
    nn = nearest_neighbor_distances(pts)
    np.testing.assert_allclose(nn, [3.0, 3.0, 4.0])


def test_nearest_neighbor_distances_degenerate() -> None:
    assert nearest_neighbor_distances(np.zeros((1, 5))).tolist() == [0.0]
    # identical points -> zero nearest-neighbour distance
    np.testing.assert_array_equal(nearest_neighbor_distances(np.zeros((3, 2))), [0, 0, 0])
