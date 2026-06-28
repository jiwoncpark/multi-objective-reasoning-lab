"""Tests for ``mobo_lab/acquisitions.py`` (the strategy-card factory)."""

from __future__ import annotations

import pytest
import torch

from mobo_lab import acquisitions, config
from mobo_lab.acquisitions import (
    PoolSelector,
    RandomSelector,
    UncertaintySelector,
    build_acquisition,
    make_sampler,
    parse_scalarized_weights,
)
from mobo_lab.models import fit_surrogate_model
from mobo_lab.pool import VHSequencePool
from mobo_lab.seed import set_all_seeds


@pytest.fixture(scope="module")
def fitted():
    set_all_seeds(0)
    X = torch.rand(12, config.LATENT_DIM, dtype=torch.double)
    Y = torch.stack([X[:, 0], 1.0 - X[:, 1]], dim=-1)
    model = fit_surrogate_model(X, Y)
    return model, X, Y


@pytest.fixture(scope="module")
def pool():
    set_all_seeds(1)
    n = 30
    X = torch.rand(n, config.LATENT_DIM, dtype=torch.double)
    return VHSequencePool(X, [f"SEQ{i}" for i in range(n)])


@pytest.mark.parametrize("name", ["nehvi", "parego", "scalarized_0.5_0.5"])
def test_botorch_acq_forward_shape(fitted, name):
    model, X, Y = fitted
    acq = build_acquisition(
        name, model, X, Y, config.REF_POINT, make_sampler(num_samples=16)
    )
    # batch_shape x q x d convention: a [1, q, d] batch scores to a single value.
    test_X = X[[0, 1, 2, 3]].unsqueeze(0)
    assert acq(test_X).shape == torch.Size([1])


def test_random_returns_pool_selector(fitted):
    model, X, Y = fitted
    acq = build_acquisition("random", model, X, Y, config.REF_POINT, make_sampler())
    assert isinstance(acq, (PoolSelector, RandomSelector))


def test_uncertainty_returns_pool_selector(fitted):
    model, X, Y = fitted
    acq = build_acquisition("uncertainty", model, X, Y, config.REF_POINT, make_sampler())
    assert isinstance(acq, (PoolSelector, UncertaintySelector))


def test_unknown_name_raises(fitted):
    model, X, Y = fitted
    with pytest.raises(KeyError, match="unknown acquisition"):
        build_acquisition("teleport", model, X, Y, config.REF_POINT, make_sampler())


@pytest.mark.parametrize(
    "name,expected",
    [
        ("scalarized_0.5_0.5", [0.5, 0.5]),
        ("scalarized_0.8_0.2", [0.8, 0.2]),
        ("scalarized_0.2_0.8", [0.2, 0.8]),
    ],
)
def test_weight_parsing(name, expected):
    assert parse_scalarized_weights(name) == expected


def test_random_selector_picks_unqueried_ids(pool):
    sel = RandomSelector(seed=0)
    observed = [0, 1, 2]
    ids = sel.select(pool, q=config.BATCH_SIZE, observed_ids=observed)
    assert len(ids) == config.BATCH_SIZE
    assert len(set(ids)) == config.BATCH_SIZE
    assert not (set(ids) & set(observed))
    # deterministic given the seed
    assert ids == RandomSelector(seed=0).select(pool, q=config.BATCH_SIZE, observed_ids=observed)


def test_random_selector_changes_with_seed(pool):
    a = RandomSelector(seed=0).select(pool, q=config.BATCH_SIZE)
    b = RandomSelector(seed=1).select(pool, q=config.BATCH_SIZE)
    assert a != b


def test_uncertainty_selector_picks_unqueried_ids(fitted, pool):
    model, X, Y = fitted
    sel = UncertaintySelector(model)
    observed = [5, 6]
    ids = sel.select(pool, q=config.BATCH_SIZE, observed_ids=observed)
    assert len(set(ids)) == config.BATCH_SIZE
    assert not (set(ids) & set(observed))


def test_strategy_names_complete():
    assert acquisitions.STRATEGY_NAMES == {
        "nehvi",
        "parego",
        "scalarized_0.5_0.5",
        "scalarized_0.8_0.2",
        "scalarized_0.2_0.8",
        "random",
        "uncertainty",
    }
