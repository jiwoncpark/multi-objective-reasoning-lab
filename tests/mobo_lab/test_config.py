"""Tests for ``mobo_lab/config.py``.

These check the invariants the rest of the lab relies on: that the reference point
matches the number of objectives, the campaign arithmetic is consistent, and the
latent-box helper has the exact shape/values BoTorch expects.
"""

from __future__ import annotations

import torch

from mobo_lab import config


def test_objective_dimensions_consistent():
    assert config.NUM_OBJECTIVES == 2
    assert len(config.REF_POINT) == config.NUM_OBJECTIVES
    assert len(config.NOISE_SIGMA) == config.NUM_OBJECTIVES
    assert config.LATENT_DIM == 5


def test_total_new_evaluations_arithmetic():
    assert config.TOTAL_NEW_EVALUATIONS == config.BATCH_SIZE * config.N_ROUNDS
    assert config.TOTAL_NEW_EVALUATIONS == 24


def test_latent_bounds_shape_and_values():
    bounds = config.latent_bounds()
    assert bounds.shape == (2, config.LATENT_DIM)
    assert bounds.dtype == torch.double
    assert torch.equal(bounds[0], torch.zeros(config.LATENT_DIM, dtype=torch.double))
    assert torch.equal(bounds[1], torch.ones(config.LATENT_DIM, dtype=torch.double))


def test_ref_point_tensor():
    rp = config.ref_point_tensor()
    assert rp.shape == (config.NUM_OBJECTIVES,)
    assert rp.dtype == torch.double
    assert rp.tolist() == config.REF_POINT


def test_data_paths_are_absolute_and_rooted():
    assert config.REPO_ROOT.is_absolute()
    assert (config.REPO_ROOT / "pyproject.toml").exists()
    for path in (
        config.LIBRARY_CSV,
        config.LATENTS_NPY,
        config.INITIAL_IDS_JSON,
        config.ORACLE_TRUE_NPY,
    ):
        assert path.is_absolute()
        assert config.DATA_DIR in path.parents
