"""Offline tests for ``scripts/build_initial_design.py`` (diverse, non-front starter set)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_initial_design as bid  # noqa: E402


def test_fps_picks_extremes_first():
    # five colinear points; starting at the left end, FPS should grab the right end next.
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    picked = bid.farthest_point_sampling(X, k=2, candidate_idx=range(5), start_idx=0)
    assert picked == [0, 4]
    # with k=3 the midpoint is added (farthest from both ends)
    picked3 = bid.farthest_point_sampling(X, k=3, candidate_idx=range(5), start_idx=0)
    assert picked3 == [0, 4, 2]


def _cloud_and_objectives(n: int = 60):
    rng = np.random.default_rng(1)
    X = rng.random((n, 5))
    Y = rng.random((n, 2))  # arbitrary objectives -> some non-dominated points
    return X, Y


def test_choose_initial_ids_count_unique_and_off_front():
    import torch

    from mobo_lab.metrics import compute_pareto_mask

    X, Y = _cloud_and_objectives()
    ids = bid.choose_initial_ids(X, Y, n_initial=12)
    assert len(ids) == 12
    assert len(set(ids)) == 12  # unique
    front = compute_pareto_mask(torch.as_tensor(Y, dtype=torch.double)).numpy()
    assert front[ids].sum() < len(ids)  # not all on the front (here: none)


def test_choose_initial_ids_deterministic():
    X, Y = _cloud_and_objectives()
    assert bid.choose_initial_ids(X, Y, n_initial=12) == bid.choose_initial_ids(X, Y, n_initial=12)


def test_choose_initial_ids_stays_in_objective_band():
    # with a large pool the band is never relaxed, so every starter sits in the
    # central-or-below region (both objectives at/below the quantile) -> low HV / headroom.
    rng = np.random.default_rng(2)
    X = rng.random((400, 5))
    Y = rng.random((400, 2))
    q = 0.6
    ids = bid.choose_initial_ids(X, Y, n_initial=12, max_quantile=q)
    assert np.all(Y[ids, 0] <= np.quantile(Y[:, 0], q))
    assert np.all(Y[ids, 1] <= np.quantile(Y[:, 1], q))
