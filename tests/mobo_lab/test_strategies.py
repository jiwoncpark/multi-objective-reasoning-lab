"""Tests for ``mobo_lab/strategies.py`` (batch plan -> validated batch of IDs)."""

from __future__ import annotations

import pytest
import torch

from mobo_lab import config
from mobo_lab.models import fit_surrogate_model
from mobo_lab.pool import VHSequencePool
from mobo_lab.seed import set_all_seeds
from mobo_lab.strategies import propose_batch_from_plan, validate_batch_plan


@pytest.fixture(scope="module")
def setup():
    set_all_seeds(0)
    n = 50
    X = torch.rand(n, config.LATENT_DIM, dtype=torch.double)
    pool = VHSequencePool(X, [f"SEQ{i}" for i in range(n)])
    observed = list(range(12))
    train_X = pool.X[observed]
    train_Y = torch.stack([train_X[:, 0], 1.0 - train_X[:, 1]], dim=-1)
    model = fit_surrogate_model(train_X, train_Y)
    return pool, model, train_X, train_Y, observed


# -- validate_batch_plan ---------------------------------------------------- #
def test_validate_rejects_wrong_sum():
    with pytest.raises(ValueError, match="sum to 3.*batch size is 4"):
        validate_batch_plan({"nehvi": 3}, batch_size=4)


def test_validate_rejects_unknown_card():
    with pytest.raises(ValueError, match="valid cards"):
        validate_batch_plan({"teleport": 4}, batch_size=4)


def test_validate_rejects_empty_plan():
    with pytest.raises(ValueError, match="empty"):
        validate_batch_plan({}, batch_size=4)


def test_validate_accepts_good_plan():
    validate_batch_plan({"nehvi": 2, "parego": 2}, batch_size=4)  # no raise


def test_validate_accepts_custom_scalarized_weights():
    # Custom fixed weights (not one of the three named cards) are valid.
    validate_batch_plan({"scalarized_0.7_0.3": 4}, batch_size=4)  # no raise


# -- propose_batch_from_plan ------------------------------------------------ #
def _check_batch(ids, observed):
    assert len(ids) == config.BATCH_SIZE
    assert len(set(ids)) == config.BATCH_SIZE  # distinct
    assert not (set(ids) & set(observed))  # none observed


@pytest.mark.parametrize("optimize", ["discrete", "continuous"])
def test_mixed_acq_plan(setup, optimize):
    pool, model, train_X, train_Y, observed = setup
    ids = propose_batch_from_plan(
        {"nehvi": 2, "parego": 2},
        model,
        pool,
        observed,
        train_X,
        train_Y,
        optimize=optimize,
    )
    _check_batch(ids, observed)


def test_random_only_plan(setup):
    pool, model, train_X, train_Y, observed = setup
    ids = propose_batch_from_plan(
        {"random": 4}, model, pool, observed, train_X, train_Y
    )
    _check_batch(ids, observed)
    avail = set(pool.available_ids(observed))
    assert set(ids) <= avail


def test_mixed_plan_with_pending_conditioning_no_repeats(setup):
    pool, model, train_X, train_Y, observed = setup
    # nehvi picks first, then random must avoid those pending IDs.
    ids = propose_batch_from_plan(
        {"nehvi": 2, "random": 2}, model, pool, observed, train_X, train_Y
    )
    _check_batch(ids, observed)


def test_uncertainty_plan(setup):
    pool, model, train_X, train_Y, observed = setup
    ids = propose_batch_from_plan(
        {"uncertainty": 2, "nehvi": 2}, model, pool, observed, train_X, train_Y
    )
    _check_batch(ids, observed)


def test_invalid_optimize_mode_raises(setup):
    pool, model, train_X, train_Y, observed = setup
    with pytest.raises(ValueError, match="discrete.*continuous"):
        propose_batch_from_plan(
            {"nehvi": 4}, model, pool, observed, train_X, train_Y, optimize="sideways"
        )
