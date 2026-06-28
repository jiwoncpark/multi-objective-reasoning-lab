"""The competition: a multi-round campaign driver, leaderboard I/O, and the reveal.

This is the multi-round wrapper around Step 9's :func:`propose_batch_from_plan`. A
team supplies a ``team_strategy`` -- one batch plan per round -- and
:func:`run_campaign` runs the fixed closed loop (fit → propose → measure → update →
record hypervolume) for ``N_ROUNDS`` rounds, enforcing the outline §10
anti-confusion rules so the contest stays fair:

* the batch size, round count, oracle, and initial design are fixed (loaded here,
  not passed by the student);
* the oracle is built with ``allow_true=False`` so no run can peek at the hidden
  objectives; and
* every round is ``BATCH_SIZE`` *distinct, never-before-measured* sequences.

Runs are scored by **AUC-HV** (area under the hypervolume-vs-round curve), with
final HV / #non-dominated selected / embedding diversity as tie-breakers. Outputs
persist to ``outputs/`` so :func:`update_leaderboard` can rank every team and
:func:`build_final_debrief_report` can draw the instructor true-front reveal.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import torch

from . import config, data, metrics, plotting
from .models import fit_surrogate_model
from .oracle import AntibodyOracle
from .pool import VHSequencePool
from .seed import set_all_seeds
from .strategies import propose_batch_from_plan, validate_batch_plan

RUN_SUFFIX = "_run.json"


def _slug(team_name: str) -> str:
    """Filesystem-safe team slug (lowercase alphanumerics + underscores)."""
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", team_name.strip()).strip("_").lower()
    return slug or "team"


# --------------------------------------------------------------------------- #
# Campaign
# --------------------------------------------------------------------------- #
def run_campaign(
    team_strategy: list[dict],
    team_name: str,
    seed: int = config.SEED,
    projection_method: str = config.PROJECTION_METHOD,
    *,
    pool: VHSequencePool | None = None,
    oracle: AntibodyOracle | None = None,
    initial_ids: list[int] | None = None,
    n_rounds: int | None = None,
    ref_point=config.REF_POINT,
    optimize: str = "discrete",
) -> dict:
    """Run a team's ``n_rounds``-round campaign and return its history dict.

    The fixed contest assets (``pool``, ``oracle``, ``initial_ids``) are loaded from
    the locked data files unless injected (tests do inject tiny fixtures). The
    oracle is always built ``allow_true=False``. ``team_strategy`` must have exactly
    ``n_rounds`` plans, each summing to ``BATCH_SIZE``.

    ``optimize`` defaults to ``"discrete"`` (the reproducible competition path);
    ``"continuous"`` runs the continuous optimizer + projection, which is where
    ``projection_method`` takes effect (used by the extensions notebook).
    """
    if pool is None:
        pool = VHSequencePool.from_files()
    if oracle is None:
        oracle = AntibodyOracle.from_files(allow_true=False)
    if initial_ids is None:
        initial_ids = data.load_initial_ids()
    if n_rounds is None:
        n_rounds = config.N_ROUNDS

    if len(team_strategy) != n_rounds:
        raise ValueError(
            f"team_strategy must have exactly {n_rounds} rounds (the fixed budget), "
            f"got {len(team_strategy)}"
        )
    for plan in team_strategy:  # fail fast before any expensive fitting
        validate_batch_plan(plan, config.BATCH_SIZE)

    set_all_seeds(seed)
    initial_ids = [int(i) for i in initial_ids]
    observed_ids = list(initial_ids)
    observed_Y = oracle.evaluate(observed_ids)

    hv_history = [metrics.compute_hypervolume(observed_Y, ref_point)]
    rounds: list[dict] = []
    selected_ids: list[int] = []

    for r, plan in enumerate(team_strategy):
        train_X = pool.X[observed_ids]
        train_Y = observed_Y
        needs_model = any(name != "random" for name in plan)
        model = fit_surrogate_model(train_X, train_Y) if needs_model else None

        new_ids = propose_batch_from_plan(
            plan, model, pool, observed_ids, train_X, train_Y,
            ref_point=ref_point, projection_method=projection_method,
            optimize=optimize, seed=seed + r,
        )
        new_Y = oracle.evaluate(new_ids)

        observed_ids = observed_ids + new_ids
        observed_Y = torch.cat([observed_Y, new_Y], dim=0)
        selected_ids.extend(new_ids)
        hv = metrics.compute_hypervolume(observed_Y, ref_point)
        hv_history.append(hv)
        rounds.append({"round": r, "plan": dict(plan), "ids": list(new_ids),
                       "Y": new_Y.tolist(), "hv": hv})

    # Anti-confusion guards (outline §10), re-checked on the assembled campaign.
    assert all(len(rd["ids"]) == config.BATCH_SIZE for rd in rounds), "fixed batch size"
    assert len(set(selected_ids)) == len(selected_ids), "no duplicate selected IDs"
    assert not (set(selected_ids) & set(initial_ids)), "never re-evaluate observed IDs"

    final_mask = metrics.compute_pareto_mask(observed_Y)
    n_nondominated = int(final_mask[len(initial_ids):].sum())

    return {
        "team_name": team_name,
        "seed": seed,
        "batch_size": config.BATCH_SIZE,
        "n_rounds": n_rounds,
        "projection_method": projection_method,
        "strategy": [dict(p) for p in team_strategy],
        "rounds": rounds,
        "hv_history": hv_history,
        "selected_ids": selected_ids,
        "final_Y": observed_Y.tolist(),
        "n_initial": len(initial_ids),
        "auc_hv": metrics.compute_auc_hv(hv_history),
        "final_hv": hv_history[-1],
        "n_nondominated_selected": n_nondominated,
        "embedding_diversity": metrics.compute_embedding_diversity(pool.X[selected_ids]),
    }


# --------------------------------------------------------------------------- #
# Persistence + leaderboard
# --------------------------------------------------------------------------- #
def save_run_outputs(history: dict, output_dir: str | Path = config.OUTPUTS_DIR) -> Path:
    """Write the run JSON, per-round CSV, and two PNGs; return the JSON path.

    Re-running the same team overwrites its files ("latest wins").
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    slug = _slug(history["team_name"])

    json_path = out / f"{slug}{RUN_SUFFIX}"
    json_path.write_text(json.dumps(history, indent=2))

    rows = [{"round": rd["round"], "plan": json.dumps(rd["plan"]),
             "ids": json.dumps(rd["ids"]), "hv": rd["hv"]} for rd in history["rounds"]]
    pd.DataFrame(rows).to_csv(out / f"{slug}_history.csv", index=False)

    final_Y = torch.tensor(history["final_Y"], dtype=torch.double)
    selected = torch.zeros(final_Y.shape[0], dtype=torch.bool)
    selected[history["n_initial"]:] = True
    ax = plotting.plot_pareto_front(
        final_Y, selected_mask=selected, ref_point=config.REF_POINT,
        title=f"{history['team_name']}: achieved front (HV {history['final_hv']:.3f})",
    )
    ax.figure.savefig(out / f"{slug}_pareto_plot.png", dpi=120, bbox_inches="tight")
    _close(ax)

    ax = plotting.plot_hv_curve(history["hv_history"], title=f"{history['team_name']}: hypervolume")
    ax.figure.savefig(out / f"{slug}_hv_curve.png", dpi=120, bbox_inches="tight")
    _close(ax)

    return json_path


def load_team_runs(output_dir: str | Path = config.OUTPUTS_DIR) -> list[dict]:
    """Load every saved ``*_run.json`` (sorted by team name) from ``output_dir``."""
    out = Path(output_dir)
    if not out.exists():
        return []
    runs = [json.loads(p.read_text()) for p in sorted(out.glob(f"*{RUN_SUFFIX}"))]
    return runs


def update_leaderboard(output_dir: str | Path = config.OUTPUTS_DIR) -> pd.DataFrame:
    """Rank all saved runs by AUC-HV, then the §10 tie-breakers."""
    runs = load_team_runs(output_dir)
    df = pd.DataFrame(
        [
            {
                "team_name": r["team_name"],
                "auc_hv": r["auc_hv"],
                "final_hv": r["final_hv"],
                "n_nondominated_selected": r["n_nondominated_selected"],
                "embedding_diversity": r["embedding_diversity"],
            }
            for r in runs
        ]
    )
    if df.empty:
        return df
    df = df.sort_values(
        by=["auc_hv", "final_hv", "n_nondominated_selected", "embedding_diversity"],
        ascending=False,
    ).reset_index(drop=True)
    df.index = df.index + 1  # 1-based rank
    df.index.name = "rank"
    return df


# --------------------------------------------------------------------------- #
# Instructor reveal
# --------------------------------------------------------------------------- #
def build_final_debrief_report(
    output_dir: str | Path,
    oracle: AntibodyOracle,
    initial_ids: list[int],
    ref_point=config.REF_POINT,
) -> Path:
    """Write the leaderboard CSV and the true-front overlay PNG; return its path.

    ``oracle`` must be built with ``allow_true=True`` (instructor only) so the
    hidden true objectives can be revealed against each team's achievement.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    leaderboard = update_leaderboard(out)
    leaderboard.to_csv(out / "leaderboard.csv")

    team_runs = load_team_runs(out)
    overlay_path = out / "final_true_pareto_overlay.png"
    ax = plotting.plot_true_front_with_team_overlays(
        oracle.true_objectives, initial_ids, team_runs, ref_point,
        output_path=overlay_path,
    )
    _close(ax)
    return overlay_path


def _close(ax) -> None:
    import matplotlib.pyplot as plt

    plt.close(ax.figure)
