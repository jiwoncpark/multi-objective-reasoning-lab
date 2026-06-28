"""The surrogate model the acquisition functions reason over.

The lab is about multi-objective *decision making*, not GP modelling, so the whole
surrogate lives behind a single call::

    model = fit_surrogate_model(train_X, train_Y)            # noise inferred
    model = fit_surrogate_model(train_X, train_Y, train_Yvar)  # known noise

Internally this is a :class:`~botorch.models.ModelListGP` of one independent
:class:`~botorch.models.SingleTaskGP` per objective -- the "objectives are modelled
independently" framing from the lecture (outline §4.3). Each sub-model normalizes
its inputs to the unit cube and standardizes its single output, so the raw latents
in ``[0, 1]`` and the roughly-``[0, 1]`` objectives are both well conditioned.

``train_Yvar`` is optional: pass per-observation noise variances (``[n, m]``) to use
the known-noise path, or leave it ``None`` to let each GP infer a homoskedastic
noise level.
"""

from __future__ import annotations

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import SumMarginalLogLikelihood
from torch import Tensor


def fit_surrogate_model(
    train_X: Tensor,
    train_Y: Tensor,
    train_Yvar: Tensor | None = None,
) -> ModelListGP:
    """Fit an independent-objective GP surrogate and return the fitted model.

    Parameters
    ----------
    train_X:
        ``[n, d]`` design points (the latent vectors of the observed sequences).
    train_Y:
        ``[n, m]`` observed objective values (maximization), one column per
        objective.
    train_Yvar:
        Optional ``[n, m]`` per-observation noise variances. ``None`` infers the
        noise level instead.

    Returns
    -------
    ModelListGP
        A fitted model list with one ``SingleTaskGP`` per objective; its
        ``posterior(X).mean`` has shape ``[..., m]``.
    """
    train_X = torch.as_tensor(train_X, dtype=torch.double)
    train_Y = torch.as_tensor(train_Y, dtype=torch.double)
    if train_X.ndim != 2 or train_Y.ndim != 2 or train_X.shape[0] != train_Y.shape[0]:
        raise ValueError(
            f"expected train_X [n, d] and train_Y [n, m] with matching n; "
            f"got {tuple(train_X.shape)} and {tuple(train_Y.shape)}"
        )
    if train_Yvar is not None:
        train_Yvar = torch.as_tensor(train_Yvar, dtype=torch.double)
        if train_Yvar.shape != train_Y.shape:
            raise ValueError(
                f"train_Yvar must match train_Y shape {tuple(train_Y.shape)}, "
                f"got {tuple(train_Yvar.shape)}"
            )

    d = train_X.shape[-1]
    models = []
    for k in range(train_Y.shape[-1]):
        yvar = None if train_Yvar is None else train_Yvar[:, k : k + 1]
        models.append(
            SingleTaskGP(
                train_X,
                train_Y[:, k : k + 1],
                train_Yvar=yvar,
                input_transform=Normalize(d=d),
                outcome_transform=Standardize(m=1),
            )
        )
    model = ModelListGP(*models)
    # With only ~12 initial points the marginal-likelihood fit may emit an
    # OptimizationWarning; fit_gpytorch_mll surfaces it but still returns a usable
    # model, which is the behaviour we want (the lab is robust to a loose fit).
    fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))
    return model
