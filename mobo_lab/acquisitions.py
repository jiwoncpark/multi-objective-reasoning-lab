"""The "strategy card" factory: build any competition acquisition by name.

:func:`build_acquisition` is the single entry point. Given a strategy ``name`` and
the fitted ``model`` plus data, it returns one of two kinds of object:

* a **BoTorch acquisition function** (``nehvi``, ``parego``, ``scalarized_*``) that
  the optimizer maximizes over the latent cube or the discrete pool; or
* a **pool selector** (``random``, ``uncertainty``) -- a tiny finite-set chooser
  that picks pool IDs directly, because these cards are not BoTorch acquisitions.

Both kinds are consumed by :mod:`mobo_lab.optimize`, which dispatches on the
:data:`PoolSelector` type. The card menu (outline §5, §12.7):

============================  ============================================================
name                          what it optimizes
============================  ============================================================
``nehvi``                     expected hypervolume improvement (the headline method)
``parego``                    random-weight Chebyshev scalarization (ParEGO)
``scalarized_0.5_0.5``        fixed balanced Chebyshev scalarization
``scalarized_0.8_0.2``        fixed objective-1-favouring scalarization
``scalarized_0.2_0.8``        fixed objective-2-favouring scalarization
``random``                    uniform pick among unqueried pool sequences (baseline)
``uncertainty``               most-uncertain unqueried pool sequences (exploration)
============================  ============================================================

The ``scalarized_*`` cards are *fixed-weight* ``qLogNParEGO``, so ParEGO and the
scalarized cards share one tested code path and differ only by random-vs-fixed
weights -- the cleanest pedagogical contrast. Chebyshev scalarization (unlike a
linear weighted sum) can still reach concave regions of the Pareto front.
"""

from __future__ import annotations

import numpy as np
import torch
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.parego import qLogNParEGO
from botorch.sampling.normal import SobolQMCNormalSampler
from torch import Tensor

from . import config


def make_sampler(
    num_samples: int = config.MC_SAMPLES, seed: int = config.SEED
) -> SobolQMCNormalSampler:
    """A seeded quasi-Monte-Carlo sampler shared by every MC acquisition.

    Seeding it (rather than relying on global RNG state) is what keeps acquisition
    values reproducible across machines, so construct one per round and hand it to
    :func:`build_acquisition`.
    """
    return SobolQMCNormalSampler(sample_shape=torch.Size([num_samples]), seed=seed)


# --------------------------------------------------------------------------- #
# Finite-set selectors (the non-BoTorch cards)
# --------------------------------------------------------------------------- #
class PoolSelector:
    """Base class for cards that pick pool IDs directly (no continuous acq).

    Subclasses implement :meth:`select`; :mod:`mobo_lab.optimize` detects this base
    type and calls it instead of running ``optimize_acqf*``.
    """

    name = "pool_selector"

    def select(
        self,
        pool,
        q: int = config.BATCH_SIZE,
        observed_ids=(),
        pending_ids=(),
    ) -> list[int]:
        raise NotImplementedError


class RandomSelector(PoolSelector):
    """Pick ``q`` unqueried pool IDs uniformly at random (the baseline card)."""

    name = "random"

    def __init__(self, seed: int = config.SEED) -> None:
        self.seed = int(seed)

    def select(self, pool, q=config.BATCH_SIZE, observed_ids=(), pending_ids=()):
        available = pool.available_ids(observed_ids, pending_ids)
        if len(available) < q:
            raise ValueError(f"only {len(available)} available IDs, need {q}")
        rng = np.random.default_rng(self.seed)
        chosen = rng.choice(len(available), size=q, replace=False)
        return [available[int(i)] for i in chosen]


class UncertaintySelector(PoolSelector):
    """Pick the ``q`` most uncertain unqueried pool sequences (exploration card).

    Scores each available pool row by its scalarized posterior standard deviation
    ``sum_k weights[k] * std_k`` and takes the top ``q`` (outline §5.4). Pure
    exploration -- it ignores the posterior mean entirely.
    """

    name = "uncertainty"

    def __init__(self, model, weights=None) -> None:
        self.model = model
        w = (1.0 / config.NUM_OBJECTIVES,) * config.NUM_OBJECTIVES if weights is None else weights
        self.weights = torch.as_tensor(w, dtype=torch.double)

    def select(self, pool, q=config.BATCH_SIZE, observed_ids=(), pending_ids=()):
        available = pool.available_ids(observed_ids, pending_ids)
        if len(available) < q:
            raise ValueError(f"only {len(available)} available IDs, need {q}")
        idx = torch.tensor(available, dtype=torch.long)
        with torch.no_grad():
            posterior = self.model.posterior(pool.X[idx])
            std = posterior.variance.clamp_min(0.0).sqrt()  # [n_avail, m]
        score = (std * self.weights).sum(dim=-1)  # [n_avail]
        top = torch.topk(score, q).indices
        return [available[int(i)] for i in top]


# --------------------------------------------------------------------------- #
# BoTorch acquisition builders
# --------------------------------------------------------------------------- #
def build_fixed_scalarized_qlognei(model, train_X: Tensor, weights, sampler):
    """Fixed-weight Chebyshev scalarization as a ``qLogNParEGO`` (outline §5.3).

    ``weights`` (one per objective, ``double``) are pinned instead of sampled, so
    the card always optimizes the *same* trade-off direction.
    """
    weights = torch.as_tensor(weights, dtype=torch.double)
    if weights.numel() != config.NUM_OBJECTIVES:
        raise ValueError(
            f"weights must have {config.NUM_OBJECTIVES} entries, got {weights.numel()}"
        )
    return qLogNParEGO(
        model=model,
        X_baseline=torch.as_tensor(train_X, dtype=torch.double),
        scalarization_weights=weights,
        sampler=sampler,
        prune_baseline=True,
    )


def parse_scalarized_weights(name: str) -> list[float]:
    """``"scalarized_0.8_0.2"`` -> ``[0.8, 0.2]`` (validates the objective count)."""
    parts = name.split("_")[1:]
    if len(parts) != config.NUM_OBJECTIVES:
        raise ValueError(
            f"scalarized name must carry {config.NUM_OBJECTIVES} weights, got {name!r}"
        )
    return [float(p) for p in parts]


def format_scalarized_name(weights) -> str:
    """``[0.8, 0.2]`` -> ``"scalarized_0.8_0.2"`` (inverse of :func:`parse_scalarized_weights`).

    Lets the extensions notebook turn a per-round weight vector into a card name
    without hand-formatting. The output round-trips through
    :func:`parse_scalarized_weights`.
    """
    weights = [float(w) for w in weights]
    if len(weights) != config.NUM_OBJECTIVES:
        raise ValueError(
            f"weights must have {config.NUM_OBJECTIVES} entries, got {len(weights)}"
        )
    return "scalarized_" + "_".join(f"{w:g}" for w in weights)


def is_known_card(name: str) -> bool:
    """True for a named card or any well-formed ``scalarized_<w1>_<w2>`` card.

    Custom fixed weights (e.g. ``scalarized_0.7_0.3``) are not in
    :data:`STRATEGY_NAMES` but are valid -- :func:`build_acquisition` builds them
    by parsing the name -- so plan validation accepts them too.
    """
    if name in STRATEGY_NAMES:
        return True
    if name.startswith("scalarized_"):
        try:
            parse_scalarized_weights(name)
            return True
        except (ValueError, IndexError):
            return False
    return False


def build_acquisition(
    name: str,
    model,
    train_X: Tensor,
    train_Y: Tensor,
    ref_point,
    sampler,
    **kwargs,
):
    """Build the acquisition (or pool selector) for strategy ``name``.

    Parameters
    ----------
    name:
        One of the cards in the module table.
    model, train_X, train_Y:
        The fitted surrogate and its training data (``train_Y`` is accepted for a
        uniform signature; the BoTorch cards read it from ``model``).
    ref_point:
        Hypervolume reference point (used by ``nehvi``).
    sampler:
        A QMC sampler from :func:`make_sampler`.
    **kwargs:
        ``seed`` for ``random``; ``weights`` for ``uncertainty``.

    Returns
    -------
    A BoTorch acquisition function, or a :class:`PoolSelector` for the finite-set
    cards (``random`` / ``uncertainty``).
    """
    train_X = torch.as_tensor(train_X, dtype=torch.double)

    if name == "nehvi":
        return qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=torch.as_tensor(ref_point, dtype=torch.double),
            X_baseline=train_X,
            sampler=sampler,
            prune_baseline=True,
        )
    if name == "parego":
        return qLogNParEGO(
            model=model,
            X_baseline=train_X,
            sampler=sampler,
            prune_baseline=True,
        )
    if name.startswith("scalarized_"):
        weights = parse_scalarized_weights(name)
        return build_fixed_scalarized_qlognei(model, train_X, weights, sampler)
    if name == "random":
        return RandomSelector(seed=int(kwargs.get("seed", config.SEED)))
    if name == "uncertainty":
        return UncertaintySelector(model, weights=kwargs.get("weights"))

    raise KeyError(f"unknown acquisition {name!r}; choices: {sorted(STRATEGY_NAMES)}")


# The full card menu, for validation and tests.
STRATEGY_NAMES = frozenset(
    {
        "nehvi",
        "parego",
        "scalarized_0.5_0.5",
        "scalarized_0.8_0.2",
        "scalarized_0.2_0.8",
        "random",
        "uncertainty",
    }
)
