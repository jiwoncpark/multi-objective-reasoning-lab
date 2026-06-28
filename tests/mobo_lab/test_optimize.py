"""Tests for ``mobo_lab/optimize.py`` (continuous + discrete batch optimizers)."""

from __future__ import annotations

import pytest
import torch

from mobo_lab import config
from mobo_lab.acquisitions import RandomSelector, build_acquisition, make_sampler
from mobo_lab.models import fit_surrogate_model
from mobo_lab.optimize import optimize_continuous, optimize_discrete
from mobo_lab.pool import VHSequencePool
from mobo_lab.seed import set_all_seeds


@pytest.fixture(scope="module")
def setup():
    set_all_seeds(0)
    n = 40
    X = torch.rand(n, config.LATENT_DIM, dtype=torch.double)
    pool = VHSequencePool(X, [f"SEQ{i}" for i in range(n)])
    observed = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    train_X = pool.X[observed]
    train_Y = torch.stack([train_X[:, 0], 1.0 - train_X[:, 1]], dim=-1)
    model = fit_surrogate_model(train_X, train_Y)
    return pool, model, train_X, train_Y, observed


def test_optimize_continuous_returns_batch_in_box(setup):
    pool, model, train_X, train_Y, observed = setup
    acq = build_acquisition(
        "nehvi", model, train_X, train_Y, config.REF_POINT, make_sampler(num_samples=16)
    )
    bounds = config.latent_bounds()
    candidates, value = optimize_continuous(acq, bounds, q=config.BATCH_SIZE)
    assert candidates.shape == (config.BATCH_SIZE, config.LATENT_DIM)
    assert float(candidates.min()) >= 0.0 and float(candidates.max()) <= 1.0


def test_optimize_discrete_returns_exact_unqueried_ids(setup):
    pool, model, train_X, train_Y, observed = setup
    acq = build_acquisition(
        "nehvi", model, train_X, train_Y, config.REF_POINT, make_sampler(num_samples=16)
    )
    candidates, ids = optimize_discrete(
        acq, pool, q=config.BATCH_SIZE, observed_ids=observed
    )
    assert len(ids) == config.BATCH_SIZE
    assert len(set(ids)) == config.BATCH_SIZE
    assert not (set(ids) & set(observed))
    # The chosen rows are exact pool rows, so candidates == pool.X[ids].
    torch.testing.assert_close(candidates, pool.X[ids])


def test_optimize_discrete_with_pool_selector(setup):
    pool, model, train_X, train_Y, observed = setup
    sel = RandomSelector(seed=0)
    candidates, ids = optimize_discrete(
        sel, pool, q=config.BATCH_SIZE, observed_ids=observed
    )
    assert len(set(ids)) == config.BATCH_SIZE
    assert not (set(ids) & set(observed))
    torch.testing.assert_close(candidates, pool.X[ids])


def test_optimize_continuous_rejects_pool_selector(setup):
    with pytest.raises(TypeError, match="finite-set selector"):
        optimize_continuous(RandomSelector(), config.latent_bounds())
