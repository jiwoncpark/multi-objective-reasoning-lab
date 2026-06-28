"""Tests for ``mobo_lab/metrics.py`` with hand-checkable numbers.

The toy objective set is the one used in Notebook 00. Hypervolumes are verified
against rectangle-area calculations done by hand.
"""

from __future__ import annotations

import torch

from mobo_lab import metrics

# Notebook 00 toy set: six candidates, two maximization objectives.
Y_TOY = torch.tensor(
    [
        [0.2, 0.8],
        [0.4, 0.6],
        [0.6, 0.4],
        [0.8, 0.2],
        [0.5, 0.5],
        [0.3, 0.3],
    ],
    dtype=torch.double,
)


def test_pareto_mask_toy():
    # Only (0.3, 0.3) is dominated -- every other point (including (0.5, 0.5),
    # which no single point beats on BOTH axes) is non-dominated.
    mask = metrics.compute_pareto_mask(Y_TOY)
    expected = torch.tensor([True, True, True, True, True, False])
    assert torch.equal(mask, expected)


def test_hypervolume_single_point():
    Y = torch.tensor([[2.0, 3.0]], dtype=torch.double)
    # rectangle [0,2] x [0,3] = 6
    assert metrics.compute_hypervolume(Y, [0.0, 0.0]) == 6.0


def test_hypervolume_two_points_union():
    Y = torch.tensor([[1.0, 3.0], [3.0, 1.0]], dtype=torch.double)
    # area(A) + area(B) - overlap = 3 + 3 - 1 = 5
    assert metrics.compute_hypervolume(Y, [0.0, 0.0]) == 5.0


def test_hypervolume_three_points_staircase():
    Y = torch.tensor([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]], dtype=torch.double)
    # vertical slices: 1*3 + 1*2 + 1*1 = 6
    assert metrics.compute_hypervolume(Y, [0.0, 0.0]) == 6.0


def test_hypervolume_monotonicity():
    base = torch.tensor([[1.0, 3.0], [3.0, 1.0]], dtype=torch.double)
    hv_base = metrics.compute_hypervolume(base, [0.0, 0.0])  # 5

    with_nondominated = torch.cat([base, torch.tensor([[2.0, 2.0]], dtype=torch.double)])
    hv_more = metrics.compute_hypervolume(with_nondominated, [0.0, 0.0])  # 6
    assert hv_more > hv_base

    with_dominated = torch.cat([base, torch.tensor([[0.5, 0.5]], dtype=torch.double)])
    hv_same = metrics.compute_hypervolume(with_dominated, [0.0, 0.0])
    assert hv_same == hv_base


def test_hypervolume_ref_point_sensitivity():
    Y = torch.tensor([[2.0, 3.0]], dtype=torch.double)
    hv_origin = metrics.compute_hypervolume(Y, [0.0, 0.0])  # 6
    hv_pessimistic = metrics.compute_hypervolume(Y, [-1.0, -1.0])  # 3*4 = 12
    assert hv_pessimistic > hv_origin
    assert hv_pessimistic == 12.0


def test_auc_hv_trapezoid():
    # trapz([0,1,2,3]) = 4.5 over 3 intervals -> mean height 1.5
    assert metrics.compute_auc_hv([0.0, 1.0, 2.0, 3.0]) == 1.5
    assert metrics.compute_auc_hv([]) == 0.0
    assert metrics.compute_auc_hv([7.0]) == 7.0


def test_embedding_diversity():
    assert metrics.compute_embedding_diversity(torch.zeros(1, 3)) == 0.0
    # two identical rows -> distance 0
    same = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.double)
    assert metrics.compute_embedding_diversity(same) == 0.0
    # 3-4-5 triangle: single pair distance 5
    pair = torch.tensor([[0.0, 0.0], [3.0, 4.0]], dtype=torch.double)
    assert metrics.compute_embedding_diversity(pair) == 5.0


def test_true_pareto_front_points_sorted():
    front = metrics.compute_true_pareto_front(Y_TOY)
    # 5 non-dominated rows, sorted by objective 1 ascending
    assert front.shape == (5, 2)
    assert torch.all(front[1:, 0] >= front[:-1, 0])
    # (0.3, 0.3) (the only dominated point) must be absent
    assert not any(torch.allclose(row, torch.tensor([0.3, 0.3], dtype=torch.double)) for row in front)
