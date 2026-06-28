"""Continuous acquisition proposal -> valid, unqueried sequence row.

BoTorch optimizes the acquisition function over the *continuous* ``[0, 1]^d``
latent cube, but the things we can actually "test in the wet lab" are the finite
set of curated VH sequences in the pool. These helpers bridge that gap: given a
batch of continuous proposals ``candidates [q, d]``, they return ``q`` distinct
pool-row indices that have not already been observed or queued (outline §3.3).

Two projection strategies are provided, both greedy and order-preserving:

* :func:`nearest` -- each proposal snaps to its closest available pool row (L2).
* :func:`diverse_nearest` -- same, but each pick is also pushed away from the
  rows already chosen *for this batch*, so a clump of similar proposals fans out
  instead of collapsing onto a handful of neighbours.

Both take a ``forbidden`` set of row indices to skip (the already-observed and
already-pending IDs). Neither mutates the caller's set -- they work on a private
copy -- and both raise ``ValueError`` if the pool runs out of available rows.

For an *exact* pool row fed in as a proposal, :func:`nearest` returns that row's
index (distance 0): the projection is then an identity lookup. This is what makes
the discrete golden path reproducible -- ``optimize_acqf_discrete`` already
returns pool rows, so snapping them back is exact (outline §3.3, docs/08).
"""

from __future__ import annotations

import torch
from torch import Tensor

# Relative weight of the diversity (repulsion) term in :func:`diverse_nearest`.
# A value of 1.0 puts "stay close to my proposal" and "stay away from the rows
# already chosen this batch" on equal footing. Larger => more spread-out batches.
DIVERSITY_WEIGHT = 1.0


def _as_candidate_matrix(candidates: Tensor, pool_X: Tensor) -> Tensor:
    """Coerce ``candidates`` to a ``[q, d]`` tensor matching ``pool_X``'s dtype."""
    cand = torch.as_tensor(candidates, dtype=pool_X.dtype)
    if cand.ndim == 1:  # a single proposal [d] -> [1, d]
        cand = cand.unsqueeze(0)
    if cand.ndim != 2 or cand.shape[1] != pool_X.shape[1]:
        raise ValueError(
            f"candidates must have shape [q, {pool_X.shape[1]}], got {tuple(cand.shape)}"
        )
    return cand


def _first_available(order: Tensor, taken: set[int]) -> int:
    """Return the first index in ``order`` (best-first) that is not in ``taken``."""
    for idx in order.tolist():
        if idx not in taken:
            return idx
    raise ValueError(
        "projection ran out of available pool rows: every candidate's neighbours "
        "are already observed, pending, or chosen this batch"
    )


def nearest(candidates: Tensor, pool_X: Tensor, forbidden: set[int]) -> list[int]:
    """Snap each proposal to its closest available pool row.

    For each row of ``candidates`` (in order), return the index of the nearest
    ``pool_X`` row (Euclidean) that is not in ``forbidden`` and has not already
    been chosen for this batch. The closest neighbour that is taken is skipped in
    favour of the next-closest, so the returned IDs are always distinct.

    Parameters
    ----------
    candidates:
        ``[q, d]`` continuous proposals (a single ``[d]`` proposal is accepted).
    pool_X:
        ``[N, d]`` pool design matrix.
    forbidden:
        Row indices to skip (already observed / pending). Not mutated.

    Returns
    -------
    list[int]
        ``q`` distinct pool-row indices, none of them in ``forbidden``.
    """
    cand = _as_candidate_matrix(candidates, pool_X)
    dists = torch.cdist(cand, pool_X)  # [q, N]
    taken = set(int(i) for i in forbidden)
    chosen: list[int] = []
    for row in range(cand.shape[0]):
        idx = _first_available(torch.argsort(dists[row]), taken)
        taken.add(idx)
        chosen.append(idx)
    return chosen


def diverse_nearest(
    candidates: Tensor,
    pool_X: Tensor,
    forbidden: set[int],
    diversity_weight: float = DIVERSITY_WEIGHT,
) -> list[int]:
    """Like :func:`nearest`, but spread the batch out.

    Each proposal still prefers nearby pool rows, but every row is also scored by
    how far it sits from the rows already chosen *for this batch*::

        score(row) = dist(row, proposal) - diversity_weight * dist(row, chosen)

    and the available row with the lowest score wins (lower distance to the
    proposal is good; larger distance to the existing picks is good). For the
    first pick ``chosen`` is empty, so this reduces exactly to :func:`nearest`.

    The result is that a clump of near-identical proposals fans out across the
    pool instead of all snapping onto the same crowded neighbourhood
    (outline §3.3, ``method="diverse_nearest"``).
    """
    cand = _as_candidate_matrix(candidates, pool_X)
    cand_dists = torch.cdist(cand, pool_X)  # [q, N]: proposal -> every pool row
    taken = set(int(i) for i in forbidden)
    chosen: list[int] = []
    for row in range(cand.shape[0]):
        if chosen:
            # Distance from every pool row to its nearest already-chosen row.
            to_chosen = torch.cdist(pool_X, pool_X[chosen]).min(dim=1).values  # [N]
            score = cand_dists[row] - diversity_weight * to_chosen
        else:
            score = cand_dists[row]
        idx = _first_available(torch.argsort(score), taken)
        taken.add(idx)
        chosen.append(idx)
    return chosen


# Dispatch table used by ``VHSequencePool.project_to_unqueried_sequences``.
METHODS = {"nearest": nearest, "diverse_nearest": diverse_nearest}
