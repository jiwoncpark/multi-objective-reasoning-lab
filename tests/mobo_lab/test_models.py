"""Tests for ``mobo_lab/models.py`` (independent-objective GP surrogate)."""

from __future__ import annotations

import pytest
import torch
from botorch.models import ModelListGP

from mobo_lab import config
from mobo_lab.models import fit_surrogate_model
from mobo_lab.seed import set_all_seeds


def _data(n: int = 12):
    set_all_seeds(0)
    X = torch.rand(n, config.LATENT_DIM, dtype=torch.double)
    # Smooth-ish objectives so the GP has something learnable.
    Y = torch.stack([X[:, 0] + 0.1 * X[:, 1], 1.0 - X[:, 2]], dim=-1)
    return X, Y


def test_fit_returns_model_list_with_right_posterior_shape():
    X, Y = _data()
    model = fit_surrogate_model(X, Y)
    assert isinstance(model, ModelListGP)
    test_X = torch.rand(5, config.LATENT_DIM, dtype=torch.double)
    post = model.posterior(test_X)
    assert post.mean.shape == (5, config.NUM_OBJECTIVES)
    assert post.variance.shape == (5, config.NUM_OBJECTIVES)


def test_fit_accepts_known_noise():
    X, Y = _data()
    Yvar = torch.full_like(Y, 0.05**2)
    model = fit_surrogate_model(X, Y, train_Yvar=Yvar)
    assert isinstance(model, ModelListGP)
    post = model.posterior(X)
    assert post.mean.shape == (X.shape[0], config.NUM_OBJECTIVES)


def test_fit_rejects_mismatched_rows():
    X = torch.rand(12, config.LATENT_DIM, dtype=torch.double)
    Y = torch.rand(10, config.NUM_OBJECTIVES, dtype=torch.double)
    with pytest.raises(ValueError, match="matching n"):
        fit_surrogate_model(X, Y)


def test_fit_rejects_mismatched_yvar():
    X, Y = _data()
    with pytest.raises(ValueError, match="train_Yvar"):
        fit_surrogate_model(X, Y, train_Yvar=torch.rand(12, 1, dtype=torch.double))
