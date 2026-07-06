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


def test_plot_pareto_front_uses_fixed_objective_window():
    # Objectives are normalized to [0, 1]; both axes should snap to the shared
    # square window so before/after figures are directly comparable.
    ax = plotting.plot_pareto_front(Y_TOY, ref_point=[-0.05, -0.05])
    assert ax.get_xlim() == plotting.OBJECTIVE_LIMS
    assert ax.get_ylim() == plotting.OBJECTIVE_LIMS
    plt.close("all")


def test_plot_pareto_front_lims_none_autoscales():
    ax = plotting.plot_pareto_front(Y_TOY, lims=None)
    assert ax.get_xlim() != plotting.OBJECTIVE_LIMS
    plt.close("all")


def test_plot_hv_curve_returns_axes():
    ax = plotting.plot_hv_curve([0.0, 1.0, 1.5, 2.0])
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_plot_true_front_with_team_overlays(tmp_path):
    Y_true_all = Y_TOY
    team_runs = [
        {"team_name": "A", "selected_ids": [0, 1], "final_hv": 0.5},
        {"team_name": "B", "selected_ids": [3], "final_hv": 0.4},
    ]
    out = tmp_path / "overlay.png"
    ax = plotting.plot_true_front_with_team_overlays(
        Y_true_all, initial_ids=[4, 5], team_runs=team_runs,
        ref_point=[-0.05, -0.05], output_path=out,
    )
    assert isinstance(ax, plt.Axes)
    assert out.exists()
    plt.close("all")
