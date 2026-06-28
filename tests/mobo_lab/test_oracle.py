"""Tests for ``mobo_lab/oracle.py`` (deterministic noisy oracle + true-objective gate)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mobo_lab import config
from mobo_lab.oracle import AntibodyOracle


def _true() -> torch.Tensor:
    return torch.tensor(
        [[0.10, 0.90], [0.50, 0.50], [0.90, 0.10], [0.30, 0.70]], dtype=torch.double
    )


def test_evaluate_shapes_and_determinism():
    oracle = AntibodyOracle(_true(), noise_sigma=(0.05, 0.05), seed=123)
    a = oracle.evaluate([0, 2, 3])
    b = oracle.evaluate([0, 2, 3])
    assert a.shape == (3, config.NUM_OBJECTIVES)
    torch.testing.assert_close(a, b)  # repeated calls identical


def test_requery_stable_per_sequence():
    oracle = AntibodyOracle(_true(), seed=7)
    # the same sequence returns the same value no matter how it is batched
    one = oracle.evaluate([1])
    twice = oracle.evaluate([1, 1])
    torch.testing.assert_close(twice[0], one[0])
    torch.testing.assert_close(twice[0], twice[1])


def test_noise_present_but_bounded():
    sigma = (0.05, 0.05)
    oracle = AntibodyOracle(_true(), noise_sigma=sigma, seed=123, allow_true=True)
    obs = oracle.evaluate([0, 1, 2, 3])
    true = oracle.true_objectives
    assert not torch.allclose(obs, true)  # noise actually added
    # but every observation is within a few sigma of the truth
    assert torch.all(torch.abs(obs - true) < 5 * torch.tensor(sigma))


def test_true_objectives_gated():
    locked = AntibodyOracle(_true(), allow_true=False)
    with pytest.raises(PermissionError):
        locked.evaluate_true([0])
    with pytest.raises(PermissionError):
        _ = locked.true_objectives

    unlocked = AntibodyOracle(_true(), allow_true=True)
    torch.testing.assert_close(unlocked.evaluate_true([0, 1]), _true()[[0, 1]])
    assert unlocked.true_objectives.shape == (4, config.NUM_OBJECTIVES)


def test_construction_validates_shape():
    with pytest.raises(ValueError, match="shape"):
        AntibodyOracle(torch.zeros(4, 3))  # 3 objectives != NUM_OBJECTIVES


def test_different_seeds_give_different_noise():
    a = AntibodyOracle(_true(), seed=1).evaluate([0, 1, 2, 3])
    b = AntibodyOracle(_true(), seed=2).evaluate([0, 1, 2, 3])
    assert not torch.allclose(a, b)


def test_from_files_roundtrip(tmp_path):
    path = tmp_path / "oracle_true_objectives.npy"
    np.save(path, _true().numpy())
    oracle = AntibodyOracle.from_files(path, allow_true=True)
    assert len(oracle) == 4
    torch.testing.assert_close(oracle.true_objectives, _true())
