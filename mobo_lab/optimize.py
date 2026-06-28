"""Turn an acquisition into a fixed-size batch of sequences to test.

Two optimizers, mirroring the two paths in the lab:

* :func:`optimize_continuous` -- the **taught headline**. Maximizes a BoTorch
  acquisition over the continuous ``[0, 1]^d`` latent cube with ``sequential=True``
  (greedy batch fill: pick candidate 1, then candidate 2 conditioned on it, ...).
  Its proposals then need :meth:`VHSequencePool.project_to_unqueried_sequences` to
  become valid sequence IDs (outline §3.3, §8.8).

* :func:`optimize_discrete` -- the **reproducible graded path**. Maximizes the
  acquisition directly over the finite pool with ``optimize_acqf_discrete``, so the
  chosen rows *are* pool rows and map back to exact IDs with no projection drift.
  This is what Notebook 01's golden constants are frozen against (outline §8.8.1).

Both also accept a :class:`~mobo_lab.acquisitions.PoolSelector` (the ``random`` /
``uncertainty`` cards): :func:`optimize_discrete` delegates to its ``select`` method,
while :func:`optimize_continuous` rejects it (those cards have no continuous form).
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from botorch.optim.optimize import optimize_acqf, optimize_acqf_discrete
from torch import Tensor

from . import config
from .acquisitions import PoolSelector


def optimize_continuous(
    acq_func,
    bounds: Tensor,
    q: int = config.BATCH_SIZE,
    num_restarts: int = config.NUM_RESTARTS,
    raw_samples: int = config.RAW_SAMPLES,
    sequential: bool = True,
) -> tuple[Tensor, Tensor]:
    """Maximize a BoTorch acquisition over the continuous latent box.

    Returns ``(candidates [q, d], acq_value)``. The candidates are continuous
    points in the box and still need projecting onto valid pool sequences.
    """
    if isinstance(acq_func, PoolSelector):
        raise TypeError(
            f"{acq_func.name!r} is a finite-set selector with no continuous form; "
            "use optimize_discrete instead"
        )
    candidates, acq_value = optimize_acqf(
        acq_function=acq_func,
        bounds=torch.as_tensor(bounds, dtype=torch.double),
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options={"batch_limit": 5, "maxiter": 200},
        sequential=sequential,
    )
    return candidates, acq_value


def optimize_discrete(
    acq_func,
    pool,
    q: int = config.BATCH_SIZE,
    observed_ids: Iterable[int] = (),
    pending_ids: Iterable[int] = (),
) -> tuple[Tensor, list[int]]:
    """Maximize the acquisition over the finite pool and return exact IDs.

    Returns ``(candidates [q, d], ids)`` where ``candidates == pool.X[ids]``. The
    returned IDs are ``q`` distinct rows, none of them observed or pending.

    For a :class:`PoolSelector` card the selector's own ``select`` does the picking;
    otherwise ``optimize_acqf_discrete`` chooses from ``pool.X`` while avoiding the
    observed and pending rows.
    """
    observed_ids = list(observed_ids)
    pending_ids = list(pending_ids)

    if isinstance(acq_func, PoolSelector):
        ids = acq_func.select(pool, q=q, observed_ids=observed_ids, pending_ids=pending_ids)
        return pool.X[ids], ids

    avoid = observed_ids + pending_ids
    X_avoid = pool.X[avoid] if avoid else None
    candidates, _acq_value = optimize_acqf_discrete(
        acq_func,
        q=q,
        choices=pool.X,
        unique=True,
        X_avoid=X_avoid,
    )
    # The chosen rows are exact pool rows, so this projection is an identity lookup
    # that also recovers the integer IDs (and re-checks distinct / unqueried).
    ids = pool.project_to_unqueried_sequences(
        candidates, observed_ids, pending_ids, method="nearest"
    )
    return candidates, ids
