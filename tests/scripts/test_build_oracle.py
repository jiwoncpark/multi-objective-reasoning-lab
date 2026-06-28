"""Offline tests for ``scripts/build_oracle.py`` (synthetic objective design)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from mobo_lab.metrics import compute_pareto_mask

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_oracle as bo  # noqa: E402


def _latent_cloud(n: int = 500) -> np.ndarray:
    """A reproducible, well-spread cloud in [0, 1]^5 to exercise the design intent."""
    rng = np.random.default_rng(0)
    return rng.random((n, 5))


def test_spearman_corr_helper():
    x = np.arange(10.0)
    assert bo.spearman_corr(x, 2 * x + 1) == 1.0  # monotone increasing
    assert bo.spearman_corr(x, -x) == -1.0  # monotone decreasing


def test_objectives_shape_range_deterministic():
    X = _latent_cloud()
    Y1 = bo.synthetic_objectives(X)
    Y2 = bo.synthetic_objectives(X)
    assert Y1.shape == (len(X), 2)
    assert Y1.min() >= 0.0 and Y1.max() <= 1.0
    np.testing.assert_array_equal(Y1, Y2)  # deterministic


def test_objectives_respond_to_their_linear_directions():
    # raw obj1 is driven by a1; a point aligned with a1 scores higher than the origin.
    a1 = np.array(bo.DEFAULT_PARAMS["a1"])
    lo = np.zeros((1, 5))
    hi = (a1 / a1.max())[None, :]  # push along the binding direction
    raw_lo = bo.synthetic_objectives_raw(lo)[0, 0]
    raw_hi = bo.synthetic_objectives_raw(hi)[0, 0]
    assert raw_hi > raw_lo


def test_near_independent_objectives():
    Y = bo.synthetic_objectives(_latent_cloud())
    rho = bo.spearman_corr(Y[:, 0], Y[:, 1])
    assert abs(rho) < 0.4  # near-independent / only a mild trade-off


def test_front_spans_separated_latent_regions():
    X = _latent_cloud()
    Y = bo.synthetic_objectives(X)
    mask = compute_pareto_mask(torch.as_tensor(Y, dtype=torch.double)).numpy()
    front_X = X[mask]
    assert mask.sum() >= 2
    # the two designed bumps sit far apart -> front points span >=2 latent regions
    pdist = np.sqrt(((front_X[:, None, :] - front_X[None, :, :]) ** 2).sum(-1))
    assert pdist.max() > 0.5
