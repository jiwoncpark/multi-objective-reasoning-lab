"""Turn a human-readable batch plan into one validated batch of sequence IDs.

A **batch plan** is a dict mapping strategy-card names to slot counts, e.g.::

    {"nehvi": 2, "parego": 2}        # 2 hypervolume picks + 2 ParEGO picks
    {"scalarized_0.8_0.2": 2, "scalarized_0.2_0.8": 2}
    {"random": 4}                    # the pure-baseline plan

:func:`propose_batch_from_plan` fills the batch one card at a time, **sequential
greedy** within each card and conditioning later cards on the IDs already chosen
(the ``pending`` set), so the whole batch is ``BATCH_SIZE`` *distinct, unqueried*
sequences -- the anti-confusion invariants from outline §10 (fixed batch size,
distinct IDs, never re-evaluate an observed sequence).

The default ``optimize="discrete"`` path is the reproducible one used by the graded
competition; ``optimize="continuous"`` runs the taught continuous optimizer and
projects its proposals onto the pool, for the syntax demos and Notebook 02.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch
from torch import Tensor

from . import config
from .acquisitions import (
    STRATEGY_NAMES,
    PoolSelector,
    build_acquisition,
    is_known_card,
    make_sampler,
)
from .optimize import optimize_continuous, optimize_discrete


def validate_batch_plan(
    batch_plan: Mapping[str, int], batch_size: int = config.BATCH_SIZE
) -> None:
    """Raise a student-friendly ``ValueError`` unless the plan is well formed.

    A valid plan is non-empty, names only known cards, has non-negative integer
    counts, and its counts sum to exactly ``batch_size``.
    """
    if not batch_plan:
        raise ValueError(
            f"batch plan is empty; it must assign {batch_size} slots across "
            f"strategy cards, e.g. {{'nehvi': {batch_size}}}."
        )
    unknown = [name for name in batch_plan if not is_known_card(name)]
    if unknown:
        raise ValueError(
            f"unknown strategy card(s) {unknown}; valid cards are {sorted(STRATEGY_NAMES)} "
            "(or any scalarized_<w1>_<w2> with custom weights)."
        )
    bad = [name for name, k in batch_plan.items() if not isinstance(k, int) or k < 0]
    if bad:
        raise ValueError(f"slot counts must be non-negative integers; bad entries: {bad}.")
    total = sum(batch_plan.values())
    if total != batch_size:
        raise ValueError(
            f"batch plan slots sum to {total}, but the batch size is {batch_size}; "
            f"adjust the counts so they add up to {batch_size}."
        )


def propose_batch_from_plan(
    batch_plan: Mapping[str, int],
    model,
    pool,
    observed_ids: Iterable[int],
    train_X: Tensor,
    train_Y: Tensor,
    ref_point=config.REF_POINT,
    bounds: Tensor | None = None,
    sampler=None,
    batch_size: int = config.BATCH_SIZE,
    projection_method: str = config.PROJECTION_METHOD,
    optimize: str = "discrete",
    seed: int = config.SEED,
) -> list[int]:
    """Fill the batch from ``batch_plan`` and return ``batch_size`` distinct IDs.

    Each card contributes its requested number of slots; later cards see the
    already-chosen IDs as ``pending`` (so a sequence is never picked twice and an
    observed sequence is never re-evaluated). Returns the combined list of IDs in
    card order.
    """
    validate_batch_plan(batch_plan, batch_size)
    if sampler is None:
        sampler = make_sampler(seed=seed)
    if bounds is None:
        bounds = config.latent_bounds()

    observed = [int(i) for i in observed_ids]
    pending: list[int] = []

    for name, k in batch_plan.items():
        if k == 0:
            continue
        acq = build_acquisition(name, model, train_X, train_Y, ref_point, sampler, seed=seed)
        if isinstance(acq, PoolSelector):
            new_ids = acq.select(pool, q=k, observed_ids=observed, pending_ids=pending)
        elif optimize == "discrete":
            _candidates, new_ids = optimize_discrete(
                acq, pool, q=k, observed_ids=observed, pending_ids=pending
            )
        elif optimize == "continuous":
            candidates, _value = optimize_continuous(acq, bounds, q=k, sequential=True)
            new_ids = pool.project_to_unqueried_sequences(
                candidates, observed, pending_ids=pending, method=projection_method
            )
        else:
            raise ValueError(
                f"optimize must be 'discrete' or 'continuous', got {optimize!r}"
            )
        pending.extend(int(i) for i in new_ids)

    # The §10 invariants, asserted before the batch leaves the helper.
    if len(pending) != batch_size:
        raise RuntimeError(
            f"batch plan produced {len(pending)} IDs, expected {batch_size}"
        )
    if len(set(pending)) != batch_size:
        raise RuntimeError(f"batch plan produced duplicate IDs: {pending}")
    if set(pending) & set(observed):
        raise RuntimeError("batch plan re-selected an already-observed sequence")
    return pending
