"""Plots for the lab: objective-space scatters, Pareto fronts, and HV curves.

Each function accepts an optional ``ax`` and returns it, never calls ``plt.show()``,
and routes all dominance logic through :mod:`mobo_lab.metrics` so the maths lives in
exactly one place. Objectives are always treated as **maximization**.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import torch

from . import metrics


def _new_ax(ax: plt.Axes | None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    return ax


def _mark_ref_point(ax: plt.Axes, ref_point) -> None:
    if ref_point is None:
        return
    ref = torch.as_tensor(ref_point, dtype=torch.double)
    ax.scatter(
        ref[0].item(),
        ref[1].item(),
        marker="x",
        s=80,
        color="black",
        label="reference point",
    )


def _pareto_staircase(front: torch.Tensor) -> tuple[list[float], list[float]]:
    """Step coordinates tracing a max/max Pareto frontier.

    ``front`` holds the non-dominated points; we sort them by objective 1 ascending
    (objective 2 then descending) and connect them with horizontal-then-vertical
    steps, matching the upper-right staircase a maximization front forms.
    """
    order = torch.argsort(front[:, 0])
    xs = front[order, 0].tolist()
    ys = front[order, 1].tolist()
    step_x: list[float] = []
    step_y: list[float] = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        if i > 0:
            step_x.append(x)
            step_y.append(step_y[-1])
        step_x.append(x)
        step_y.append(y)
    return step_x, step_y


def plot_objective_space(
    Y: torch.Tensor,
    pareto_mask: torch.Tensor | None = None,
    ref_point=None,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Scatter objective 1 vs objective 2, optionally highlighting non-dominated points."""
    ax = _new_ax(ax)
    if pareto_mask is None:
        ax.scatter(Y[:, 0], Y[:, 1], color="tab:blue", label="candidates")
    else:
        dom = ~pareto_mask
        ax.scatter(Y[dom, 0], Y[dom, 1], color="lightgray", label="dominated")
        ax.scatter(
            Y[pareto_mask, 0],
            Y[pareto_mask, 1],
            color="tab:red",
            label="non-dominated",
        )
    _mark_ref_point(ax, ref_point)
    ax.set_xlabel("objective 1 (higher is better)")
    ax.set_ylabel("objective 2 (higher is better)")
    if title:
        ax.set_title(title)
    ax.legend(loc="best", fontsize="small")
    return ax


def plot_pareto_front(
    Y: torch.Tensor,
    selected_mask: torch.Tensor | None = None,
    ref_point=None,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Scatter all points, draw the Pareto frontier, and optionally flag selected points."""
    ax = _new_ax(ax)
    ax.scatter(Y[:, 0], Y[:, 1], color="lightgray", label="all points")

    mask = metrics.compute_pareto_mask(Y)
    front = Y[mask]
    if front.shape[0] >= 1:
        step_x, step_y = _pareto_staircase(front)
        ax.plot(step_x, step_y, color="tab:red", linewidth=1.5, zorder=2)
        ax.scatter(
            front[:, 0], front[:, 1], color="tab:red", zorder=3, label="Pareto front"
        )

    if selected_mask is not None and bool(selected_mask.any()):
        ax.scatter(
            Y[selected_mask, 0],
            Y[selected_mask, 1],
            facecolors="none",
            edgecolors="tab:green",
            s=140,
            linewidths=1.8,
            zorder=4,
            label="newly selected",
        )

    _mark_ref_point(ax, ref_point)
    ax.set_xlabel("objective 1 (higher is better)")
    ax.set_ylabel("objective 2 (higher is better)")
    if title:
        ax.set_title(title)
    ax.legend(loc="best", fontsize="small")
    return ax


def plot_hv_curve(
    hv_history: Sequence[float],
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot hypervolume against round index."""
    ax = _new_ax(ax)
    rounds = list(range(len(hv_history)))
    ax.plot(rounds, list(hv_history), marker="o", color="tab:blue")
    ax.set_xlabel("round")
    ax.set_ylabel("hypervolume")
    if title:
        ax.set_title(title)
    return ax


def plot_true_front_with_team_overlays(
    Y_true_all: torch.Tensor,
    initial_ids: Sequence[int],
    team_runs: Sequence[dict],
    ref_point,
    output_path=None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """The instructor reveal: true Pareto front with each team's achievement on top.

    Layers (outline §10): all candidates in *true* objective space (gray), the true
    Pareto front (black staircase), the shared initial design (outlined), and per
    team -- their selected points (faint) and their achieved non-dominated front
    (colored). ``team_runs`` are the history dicts from
    :func:`mobo_lab.competition.load_team_runs` (each needs ``team_name`` and
    ``selected_ids``). If ``output_path`` is given the figure is saved there.
    """
    ax = _new_ax(ax)
    Y = torch.as_tensor(Y_true_all, dtype=torch.double)
    initial_ids = [int(i) for i in initial_ids]

    ax.scatter(Y[:, 0], Y[:, 1], color="lightgray", s=12, zorder=1, label="all candidates (true)")

    front = metrics.compute_true_pareto_front(Y)
    step_x, step_y = _pareto_staircase(front)
    ax.plot(step_x, step_y, color="black", linewidth=2.0, zorder=3, label="true Pareto front")
    ax.scatter(front[:, 0], front[:, 1], color="black", s=30, zorder=3)

    init = Y[initial_ids]
    ax.scatter(
        init[:, 0], init[:, 1], facecolors="none", edgecolors="dimgray",
        s=90, linewidths=1.5, zorder=4, label="initial design",
    )

    palette = plt.get_cmap("tab10").colors
    for i, run in enumerate(team_runs):
        color = palette[i % len(palette)]
        selected = [int(j) for j in run["selected_ids"]]
        name = run.get("team_name", f"team {i}")
        if selected:
            sel_Y = Y[selected]
            ax.scatter(sel_Y[:, 0], sel_Y[:, 1], color=color, s=18, alpha=0.25, zorder=2)
        team_Y = Y[initial_ids + selected]
        team_front = metrics.compute_true_pareto_front(team_Y)
        fx, fy = _pareto_staircase(team_front)
        label = f"{name} (HV {run.get('final_hv', float('nan')):.3f})"
        ax.plot(fx, fy, color=color, linewidth=1.5, zorder=5, label=label)
        ax.scatter(team_front[:, 0], team_front[:, 1], color=color, s=40, zorder=5)

    _mark_ref_point(ax, ref_point)
    ax.set_xlabel("objective 1 (true, higher is better)")
    ax.set_ylabel("objective 2 (true, higher is better)")
    ax.set_title("True Pareto front vs team achievements")
    ax.legend(loc="best", fontsize="x-small")
    if output_path is not None:
        ax.figure.savefig(output_path, dpi=120, bbox_inches="tight")
    return ax
