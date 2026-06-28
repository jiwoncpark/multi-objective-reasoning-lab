"""Multi-objective bookkeeping: Pareto masks, hypervolume, and campaign scores.

These are thin, well-tested wrappers over BoTorch's multi-objective utilities so the
notebooks can treat them as a trustworthy black box. Everything assumes
**maximization** of every objective and ``torch.double`` tensors of shape ``[n, m]``
(``n`` points, ``m`` objectives).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import is_non_dominated


def compute_pareto_mask(Y: torch.Tensor) -> torch.Tensor:
    """Boolean mask ``[n]`` marking the non-dominated (Pareto-optimal) rows of ``Y``.

    A point is non-dominated if no other point is greater-or-equal on every
    objective and strictly greater on at least one (maximization).
    """
    return is_non_dominated(Y)


def compute_hypervolume(Y: torch.Tensor, ref_point) -> float:
    """Hypervolume dominated by ``Y`` relative to ``ref_point`` (maximization).

    ``ref_point`` may be a list or a tensor; points that do not dominate the
    reference contribute nothing.
    """
    ref = torch.as_tensor(ref_point, dtype=Y.dtype)
    partitioning = DominatedPartitioning(ref_point=ref, Y=Y)
    return partitioning.compute_hypervolume().item()


def compute_auc_hv(hv_history: Sequence[float]) -> float:
    """Time-averaged area under the hypervolume-vs-round curve.

    Trapezoidal integral of ``hv_history`` over the round index, divided by the
    number of intervals, so the score is the *mean* hypervolume across the
    campaign and is comparable between runs with the same number of rounds.
    Returns ``0.0`` for an empty history and the single value for a length-1 history.
    """
    n = len(hv_history)
    if n == 0:
        return 0.0
    if n == 1:
        return float(hv_history[0])
    area = sum((hv_history[i] + hv_history[i + 1]) / 2.0 for i in range(n - 1))
    return area / (n - 1)


def compute_embedding_diversity(X_selected: torch.Tensor) -> float:
    """Mean pairwise Euclidean distance among the selected latent points.

    Returns ``0.0`` when fewer than two points are given (no pairs).
    """
    if X_selected.shape[0] < 2:
        return 0.0
    return torch.pdist(X_selected).mean().item()


def compute_true_pareto_front(Y_true_all: torch.Tensor) -> torch.Tensor:
    """Non-dominated rows of the hidden true-objective table, sorted for plotting.

    Returns the Pareto-optimal points themselves (shape ``[k, m]``), sorted by the
    first objective ascending so they can be drawn as a clean frontier. Used only
    by the instructor reveal.
    """
    mask = compute_pareto_mask(Y_true_all)
    front = Y_true_all[mask]
    order = torch.argsort(front[:, 0])
    return front[order]
