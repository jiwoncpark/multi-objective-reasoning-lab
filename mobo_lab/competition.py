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
class Campaign:
    """Stateful, one-round-at-a-time campaign so teams can *adapt* between rounds.

    Play a single round with :meth:`play_round`, read the hypervolume it reports,
    then decide the *next* round's plan in light of what you just saw -- the whole
    point of an adaptive campaign. When all ``n_rounds`` rounds are played, call
    :meth:`finalize` to get the same history dict :func:`run_campaign` returns (feed
    it straight to :func:`save_run_outputs`).

    Reproducibility is preserved. Each round's randomness is seeded by
    ``seed + round_index`` (via :func:`propose_batch_from_plan`), and the surrogate
    fit is deterministic, so a given *sequence of plans* reproduces exactly whether
    played interactively here or in one shot through :func:`run_campaign` -- and
    inspecting hypervolume between rounds (pure lookups, no RNG) cannot perturb it.
    The same outline §10 fairness rules are enforced every round: fixed
    ``BATCH_SIZE``, no duplicates, never re-measure an already-observed antibody.

    The fixed contest assets (``pool``, ``oracle``, ``initial_ids``) load from the
    locked data files unless injected (tests pass tiny fixtures). The oracle is built
    ``allow_true=False`` so a running campaign can never peek at the hidden objectives.
    """

    def __init__(
        self,
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
    ) -> None:
        self.team_name = team_name
        self.seed = seed
        self.projection_method = projection_method
        self.ref_point = ref_point
        self.optimize = optimize
        self._pool = pool if pool is not None else VHSequencePool.from_files()
        self._oracle = (
            oracle if oracle is not None else AntibodyOracle.from_files(allow_true=False)
        )
        resolved_initial = initial_ids if initial_ids is not None else data.load_initial_ids()
        self.n_rounds = n_rounds if n_rounds is not None else config.N_ROUNDS

        set_all_seeds(seed)
        self._initial_ids = [int(i) for i in resolved_initial]
        self._observed_ids = list(self._initial_ids)
        self._observed_Y = self._oracle.evaluate(self._observed_ids)
        self._hv_history = [metrics.compute_hypervolume(self._observed_Y, ref_point)]
        self._rounds: list[dict] = []
        self._selected_ids: list[int] = []

    # -- state ------------------------------------------------------------- #
    @property
    def round_index(self) -> int:
        """How many rounds have been played so far (0 before the first)."""
        return len(self._rounds)

    @property
    def rounds_left(self) -> int:
        """Rounds still to play before the campaign is complete."""
        return self.n_rounds - self.round_index

    @property
    def current_hv(self) -> float:
        """Hypervolume of everything observed so far (initial design + all rounds)."""
        return self._hv_history[-1]

    # -- play -------------------------------------------------------------- #
    def play_round(self, plan: dict, *, verbose: bool = True) -> dict:
        """Play one round from ``plan``; update state and return that round's record.

        Fits the surrogate on everything observed so far, proposes ``BATCH_SIZE``
        distinct new antibodies per ``plan``, "measures" them, and records the new
        hypervolume. Prints a one-line ``HV before -> after (+gain)`` summary unless
        ``verbose=False`` -- read it, then choose the next round's plan.
        """
        if self.round_index >= self.n_rounds:
            raise RuntimeError(
                f"all {self.n_rounds} rounds already played; call finalize() to score "
                "the campaign"
            )
        validate_batch_plan(plan, config.BATCH_SIZE)

        r = self.round_index
        train_X = self._pool.X[self._observed_ids]
        train_Y = self._observed_Y
        needs_model = any(name != "random" for name in plan)
        model = fit_surrogate_model(train_X, train_Y) if needs_model else None

        new_ids = propose_batch_from_plan(
            plan, model, self._pool, self._observed_ids, train_X, train_Y,
            ref_point=self.ref_point, projection_method=self.projection_method,
            optimize=self.optimize, seed=self.seed + r,
        )
        # Per-round fairness guards (outline §10), checked before we commit the round.
        assert len(new_ids) == config.BATCH_SIZE, "fixed batch size"
        assert len(set(new_ids)) == len(new_ids), "no duplicate selected IDs"
        assert not (set(new_ids) & set(self._observed_ids)), "never re-evaluate observed IDs"

        new_Y = self._oracle.evaluate(new_ids)
        hv_before = self._hv_history[-1]
        self._observed_ids = self._observed_ids + new_ids
        self._observed_Y = torch.cat([self._observed_Y, new_Y], dim=0)
        self._selected_ids.extend(new_ids)
        hv = metrics.compute_hypervolume(self._observed_Y, self.ref_point)
        self._hv_history.append(hv)
        record = {"round": r, "plan": dict(plan), "ids": list(new_ids),
                  "Y": new_Y.tolist(), "hv": hv}
        self._rounds.append(record)

        if verbose:
            print(
                f"round {r + 1}/{self.n_rounds}  "
                f"HV {hv_before:.4f} -> {hv:.4f}  (+{hv - hv_before:.4f})  "
                f"selected {new_ids}"
            )
        return record

    def finalize(self) -> dict:
        """Return the scored history dict (needs all ``n_rounds`` rounds played)."""
        if self.round_index != self.n_rounds:
            raise RuntimeError(
                f"played {self.round_index} of {self.n_rounds} rounds; play "
                f"{self.rounds_left} more before finalize()"
            )
        # Anti-confusion guards (outline §10), re-checked on the assembled campaign.
        assert all(len(rd["ids"]) == config.BATCH_SIZE for rd in self._rounds), "fixed batch size"
        assert len(set(self._selected_ids)) == len(self._selected_ids), "no duplicate selected IDs"
        assert not (set(self._selected_ids) & set(self._initial_ids)), "never re-evaluate observed IDs"

        final_mask = metrics.compute_pareto_mask(self._observed_Y)
        n_nondominated = int(final_mask[len(self._initial_ids):].sum())
        return {
            "team_name": self.team_name,
            "seed": self.seed,
            "batch_size": config.BATCH_SIZE,
            "n_rounds": self.n_rounds,
            "projection_method": self.projection_method,
            "strategy": [dict(rd["plan"]) for rd in self._rounds],
            "rounds": self._rounds,
            "hv_history": self._hv_history,
            "selected_ids": self._selected_ids,
            "final_Y": self._observed_Y.tolist(),
            "n_initial": len(self._initial_ids),
            "auc_hv": metrics.compute_auc_hv(self._hv_history),
            "final_hv": self._hv_history[-1],
            "n_nondominated_selected": n_nondominated,
            "embedding_diversity": metrics.compute_embedding_diversity(
                self._pool.X[self._selected_ids]
            ),
        }

    # -- visualization (used by the notebook between rounds) --------------- #
    def plot_front(self, ax=None):
        """Objective-space scatter of the achieved front so far (selected flagged)."""
        selected = torch.zeros(self._observed_Y.shape[0], dtype=torch.bool)
        selected[len(self._initial_ids):] = True
        return plotting.plot_pareto_front(
            self._observed_Y, selected_mask=selected, ref_point=self.ref_point,
            title=f"{self.team_name}: front after round {self.round_index} "
                  f"(HV {self.current_hv:.3f})",
            ax=ax,
        )

    def plot_hv(self, ax=None):
        """Hypervolume-vs-round curve for the rounds played so far."""
        return plotting.plot_hv_curve(
            self._hv_history, title=f"{self.team_name}: hypervolume vs round", ax=ax
        )

    def plot_fronts_by_round(self, ax=None):
        """Overlay the achieved Pareto front after each round played, colored by round."""
        n0 = len(self._initial_ids)
        initial_Y = self._observed_Y[:n0]
        round_Ys = [torch.as_tensor(rd["Y"], dtype=torch.double) for rd in self._rounds]
        round_hvs = [rd["hv"] for rd in self._rounds]
        ax = plotting.plot_fronts_by_round(
            initial_Y, round_Ys, ref_point=self.ref_point, round_hvs=round_hvs, ax=ax,
        )
        ax.set_title(f"{self.team_name}: Pareto front by round")
        return ax


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

    This is the one-shot (open-loop) path: it plays a *pre-committed* strategy. For
    the interactive path where a team adapts each round from the observed
    hypervolume, drive :class:`Campaign` directly -- both share the same per-round
    machinery, so the same sequence of plans yields an identical history.
    """
    campaign = Campaign(
        team_name, seed=seed, projection_method=projection_method,
        pool=pool, oracle=oracle, initial_ids=initial_ids, n_rounds=n_rounds,
        ref_point=ref_point, optimize=optimize,
    )
    if len(team_strategy) != campaign.n_rounds:
        raise ValueError(
            f"team_strategy must have exactly {campaign.n_rounds} rounds (the fixed "
            f"budget), got {len(team_strategy)}"
        )
    for plan in team_strategy:  # fail fast before any expensive fitting
        validate_batch_plan(plan, config.BATCH_SIZE)

    for plan in team_strategy:
        campaign.play_round(plan, verbose=False)
    return campaign.finalize()


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
