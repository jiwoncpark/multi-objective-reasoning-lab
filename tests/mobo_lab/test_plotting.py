"""Smoke tests for ``mobo_lab/plotting.py``.

We render headlessly (Agg) and only assert each function returns an ``Axes`` without
error; the dominance/HV maths is covered in ``test_metrics.py``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # noqa: E402  (must precede pyplot import)
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from mobo_lab import metrics, plotting  # noqa: E402

Y_TOY = torch.tensor(
    [[0.2, 0.8], [0.4, 0.6], [0.6, 0.4], [0.8, 0.2], [0.5, 0.5], [0.3, 0.3]],
    dtype=torch.double,
)


def test_plot_objective_space_returns_axes():
    ax = plotting.plot_objective_space(Y_TOY)
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_plot_objective_space_with_mask_and_ref():
    mask = metrics.compute_pareto_mask(Y_TOY)
    ax = plotting.plot_objective_space(Y_TOY, pareto_mask=mask, ref_point=[0.0, 0.0])
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_plot_pareto_front_with_selection():
    selected = torch.zeros(Y_TOY.shape[0], dtype=torch.bool)
    selected[0] = True
    ax = plotting.plot_pareto_front(Y_TOY, selected_mask=selected, ref_point=[-0.05, -0.05])
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_plot_hv_curve_returns_axes():
    ax = plotting.plot_hv_curve([0.0, 1.0, 1.5, 2.0])
    assert isinstance(ax, plt.Axes)
    plt.close("all")
