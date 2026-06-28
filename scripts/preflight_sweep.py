"""Instructor preflight: is this a competition worth running? (docs/11, GATE)

Before any golden values are frozen (Step 11) or the lab is taught, this script
sweeps the candidate strategies under the *same* initial design, budget, and batch
size on the real assets and checks the outline §18.4 acceptance criteria:

* strategies actually diverge,
* model-guided (``nehvi``) beats the ``random`` baseline on average,
* fixed scalarization concentrates in one objective-space region while ParEGO
  spreads across trade-offs,
* the leaderboard is not predetermined by one obvious strategy, and
* a full campaign runs comfortably within the practicum schedule.

It also checks the **discrete acquisition margin** in the golden-path round (the
4th-vs-5th best pool point), the float-level robustness that keeps Notebook 01
reproducible across machines.

This is an *instructor* tool: it writes a console PASS/FAIL banner and a few PNGs
under ``outputs/preflight/`` and is never imported by the student notebooks. The
campaign loop here is local so the preflight can run ahead of the Step 13
``competition.run_campaign`` (which will later supersede it).

Usage
-----
    python scripts/preflight_sweep.py                 # full sweep on the real pool
    python scripts/preflight_sweep.py --seeds 5       # average over more seeds
    python scripts/preflight_sweep.py --oracle-params data/oracle_params.json
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from mobo_lab import config, metrics, plotting  # noqa: E402
from mobo_lab.acquisitions import build_acquisition, make_sampler  # noqa: E402
from mobo_lab.models import fit_surrogate_model  # noqa: E402
from mobo_lab.oracle import AntibodyOracle  # noqa: E402
from mobo_lab.pool import VHSequencePool  # noqa: E402
from mobo_lab.seed import set_all_seeds  # noqa: E402
from mobo_lab.strategies import propose_batch_from_plan  # noqa: E402


# --------------------------------------------------------------------------- #
# Campaign loop (local; superseded by competition.run_campaign in Step 13)
# --------------------------------------------------------------------------- #
def run_campaign(
    strategy_rounds: list[dict[str, int]],
    pool: VHSequencePool,
    oracle: AntibodyOracle,
    initial_ids: list[int],
    ref_point=config.REF_POINT,
    seed: int = config.SEED,
) -> dict:
    """Run one closed-loop campaign and return its trajectory.

    Starts from ``initial_ids`` (shared by all strategies), then for each round's
    batch plan: fit the surrogate (skipped for pure-``random`` rounds), propose a
    batch, query the oracle, and record the hypervolume. Returns a dict with the
    HV history, the newly selected IDs, and their observed objectives.
    """
    set_all_seeds(seed)
    observed_ids = list(initial_ids)
    observed_Y = oracle.evaluate(observed_ids)

    hv_history = [metrics.compute_hypervolume(observed_Y, ref_point)]
    selected_ids: list[int] = []

    for r, plan in enumerate(strategy_rounds):
        train_X = pool.X[observed_ids]
        train_Y = observed_Y
        needs_model = any(name != "random" for name in plan)
        model = fit_surrogate_model(train_X, train_Y) if needs_model else None

        new_ids = propose_batch_from_plan(
            plan,
            model,
            pool,
            observed_ids,
            train_X,
            train_Y,
            ref_point=ref_point,
            seed=seed + r,  # per-round, reproducible randomness (ParEGO / random card)
        )
        new_Y = oracle.evaluate(new_ids)

        observed_ids = observed_ids + new_ids
        observed_Y = torch.cat([observed_Y, new_Y], dim=0)
        selected_ids.extend(new_ids)
        hv_history.append(metrics.compute_hypervolume(observed_Y, ref_point))

    return {
        "hv_history": hv_history,
        "selected_ids": selected_ids,
        "selected_Y": observed_Y[len(initial_ids):],
        "final_Y": observed_Y,
    }


# --------------------------------------------------------------------------- #
# Per-campaign metrics
# --------------------------------------------------------------------------- #
def _selection_angles(selected_Y: torch.Tensor, ref_point) -> torch.Tensor:
    """Angle (radians) of each selected point relative to the reference point.

    With both objectives maximized and above the reference, the angle in the
    objective plane encodes *which trade-off region* a point sits in. A strategy
    that concentrates on one region produces a tight band of angles; one that
    explores many trade-offs produces a wide spread.
    """
    ref = torch.as_tensor(ref_point, dtype=torch.double)
    offset = (selected_Y - ref).clamp_min(1e-9)
    return torch.atan2(offset[:, 1], offset[:, 0])


def angular_spread(selected_Y: torch.Tensor, ref_point) -> float:
    """Std-dev of selection angles -- small = concentrated, large = exploratory."""
    if selected_Y.shape[0] < 2:
        return 0.0
    return float(_selection_angles(selected_Y, ref_point).std())


def region_coverage(selected_Y: torch.Tensor, ref_point, n_bins: int = 6) -> int:
    """How many of ``n_bins`` angular regions (0..pi/2) the selection touches."""
    if selected_Y.shape[0] == 0:
        return 0
    angles = _selection_angles(selected_Y, ref_point).clamp(0.0, math.pi / 2 - 1e-9)
    bins = (angles / (math.pi / 2) * n_bins).floor().long()
    return int(torch.unique(bins).numel())


def campaign_metrics(result: dict, ref_point=config.REF_POINT, pool=None) -> dict:
    """Reduce a campaign trajectory to the §18.1 scalar metrics."""
    hv = result["hv_history"]
    selected_Y = result["selected_Y"]
    final_Y = result["final_Y"]

    # How many *selected* points end up on the final observed Pareto front.
    final_mask = metrics.compute_pareto_mask(final_Y)
    n_initial = final_Y.shape[0] - selected_Y.shape[0]
    n_selected_nondom = int(final_mask[n_initial:].sum())

    diversity = 0.0
    if pool is not None and result["selected_ids"]:
        diversity = metrics.compute_embedding_diversity(pool.X[result["selected_ids"]])

    return {
        "auc_hv": metrics.compute_auc_hv(hv),
        "final_hv": hv[-1],
        "hv_gain_per_round": [hv[i + 1] - hv[i] for i in range(len(hv) - 1)],
        "n_selected_nondominated": n_selected_nondom,
        "selected_diversity": diversity,
        "angular_spread": angular_spread(selected_Y, ref_point),
        "region_coverage": region_coverage(selected_Y, ref_point),
    }


# --------------------------------------------------------------------------- #
# Strategy menu (outline §18.1)
# --------------------------------------------------------------------------- #
def default_strategies(n_rounds: int = config.N_ROUNDS) -> dict[str, list[dict]]:
    """The §18.1 candidate strategies, padded/truncated to ``n_rounds`` rounds."""

    def pad(rounds: list[dict]) -> list[dict]:
        if len(rounds) >= n_rounds:
            return rounds[:n_rounds]
        return rounds + [rounds[-1]] * (n_rounds - len(rounds))

    return {
        "all_nehvi": pad([{"nehvi": 4}]),
        "all_parego": pad([{"parego": 4}]),
        "all_scalarized_0.8_0.2": pad([{"scalarized_0.8_0.2": 4}]),
        "scalarization_sweep": pad(
            [
                {"scalarized_0.8_0.2": 2, "scalarized_0.2_0.8": 2},
                {"scalarized_0.8_0.2": 2, "scalarized_0.2_0.8": 2},
                {"scalarized_0.5_0.5": 4},
                {"scalarized_0.5_0.5": 4},
                {"nehvi": 4},
                {"nehvi": 4},
            ]
        ),
        "explore_then_exploit": pad(
            [
                {"random": 2, "parego": 2},
                {"parego": 4},
                {"nehvi": 2, "parego": 2},
                {"nehvi": 4},
            ]
        ),
        "mixed": pad(
            [
                {"nehvi": 2, "parego": 2},
                {"nehvi": 3, "random": 1},
                {"scalarized_0.8_0.2": 2, "scalarized_0.2_0.8": 2},
                {"nehvi": 2, "parego": 2},
                {"nehvi": 4},
            ]
        ),
        "random_baseline": pad([{"random": 4}]),
    }


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #
def run_sweep(
    strategies: dict[str, list[dict]],
    pool: VHSequencePool,
    oracle: AntibodyOracle,
    initial_ids: list[int],
    seeds: list[int],
    ref_point=config.REF_POINT,
) -> dict[str, dict]:
    """Run every strategy across every seed; return per-strategy aggregated metrics."""
    out: dict[str, dict] = {}
    for name, rounds in strategies.items():
        per_seed = []
        for s in seeds:
            result = run_campaign(rounds, pool, oracle, initial_ids, ref_point, seed=s)
            per_seed.append(campaign_metrics(result, ref_point, pool=pool))
        out[name] = {
            "per_seed": per_seed,
            "auc_hv_mean": _mean(m["auc_hv"] for m in per_seed),
            "auc_hv_by_seed": [m["auc_hv"] for m in per_seed],
            "final_hv_mean": _mean(m["final_hv"] for m in per_seed),
            "angular_spread_mean": _mean(m["angular_spread"] for m in per_seed),
            "region_coverage_mean": _mean(m["region_coverage"] for m in per_seed),
            "diversity_mean": _mean(m["selected_diversity"] for m in per_seed),
            "n_nondom_mean": _mean(m["n_selected_nondominated"] for m in per_seed),
        }
    return out


def _mean(values) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


# --------------------------------------------------------------------------- #
# Discrete acquisition margin (golden-path robustness)
# --------------------------------------------------------------------------- #
def discrete_acq_margin(
    pool: VHSequencePool,
    oracle: AntibodyOracle,
    initial_ids: list[int],
    q: int = config.BATCH_SIZE,
    seed: int = config.SEED,
) -> dict:
    """Margin between the qth- and (q+1)th-best single-point ``nehvi`` values.

    Evaluates ``nehvi`` on every *available* pool row as a one-point batch and
    sorts. A clear gap at the q/(q+1) boundary means the discrete argmax is robust
    to float-level kernel differences across machines (the qLogNEHVI fused-vs-pure
    caveat), protecting Notebook 01's reproducibility.
    """
    set_all_seeds(seed)
    train_X = pool.X[initial_ids]
    train_Y = oracle.evaluate(initial_ids)
    model = fit_surrogate_model(train_X, train_Y)
    acq = build_acquisition(
        "nehvi", model, train_X, train_Y, config.REF_POINT, make_sampler(seed=seed)
    )
    available = pool.available_ids(initial_ids)
    X_avail = pool.X[available]
    with torch.no_grad():
        values = acq(X_avail.unsqueeze(1))  # [n_avail, 1, d] -> [n_avail]
    top = torch.sort(values, descending=True).values
    qth, nextth = float(top[q - 1]), float(top[q])
    spread = float(top[0] - top[-1]) or 1.0
    return {
        "qth_value": qth,
        "next_value": nextth,
        "abs_margin": qth - nextth,
        "rel_margin": (qth - nextth) / abs(spread),
    }


# --------------------------------------------------------------------------- #
# §18.4 acceptance criteria
# --------------------------------------------------------------------------- #
class Criterion:
    def __init__(self, name: str, passed: bool, detail: str, hint: str = "", info: bool = False):
        self.name = name
        self.passed = passed
        self.detail = detail
        self.hint = hint
        self.info = info


def evaluate_criteria(
    sweep: dict[str, dict],
    determinism_ok: bool,
    margin: dict,
    campaign_seconds: float,
    plots_written: int,
    reveal: dict,
) -> list[Criterion]:
    """Turn the sweep + checks into the §18.4 PASS/FAIL list."""
    crit: list[Criterion] = []

    crit.append(
        Criterion(
            "1. Golden-path inputs deterministic",
            determinism_ok,
            "re-running a campaign with the same seed reproduced HV + IDs exactly",
            hint="non-determinism upstream; check seeding in run_campaign / strategies",
        )
    )

    nehvi = sweep["all_nehvi"]["auc_hv_mean"]
    rnd = sweep["random_baseline"]["auc_hv_mean"]
    crit.append(
        Criterion(
            "2. nehvi beats random (AUC-HV)",
            nehvi > rnd,
            f"all_nehvi {nehvi:.4f} vs random_baseline {rnd:.4f}",
            hint="weak latent signal: strengthen oracle structure (Step 6) or embedding spread (Step 5)",
        )
    )

    # 3. some non-pure-nehvi strategy beats all_nehvi in at least one seed.
    nehvi_by_seed = sweep["all_nehvi"]["auc_hv_by_seed"]
    challengers = ["mixed", "scalarization_sweep", "explore_then_exploit", "all_parego"]
    beats = {
        name: [
            sweep[name]["auc_hv_by_seed"][i] > nehvi_by_seed[i]
            for i in range(len(nehvi_by_seed))
        ]
        for name in challengers
    }
    any_beats = any(any(v) for v in beats.values())
    winners_detail = ", ".join(f"{n}:{sum(v)}/{len(v)}" for n, v in beats.items())
    crit.append(
        Criterion(
            "3. a mixed strategy sometimes beats all_nehvi",
            any_beats,
            f"per-seed wins over all_nehvi -> {winners_detail}",
            hint="all_nehvi too dominant: add isolated front regions (Step 6 bumps) so exploration pays off",
        )
    )

    # Criteria 4 & 5 are scored on *region coverage* (how many objective-space
    # regions a strategy's selections touch), not angular spread. Coverage is the
    # stable discriminator across seeds; angular spread of the selected points is
    # too noisy to separate fixed-weight from random-weight ParEGO (their selected
    # points sit at similar angles). Angular spread is still reported for context.
    scal_cov = sweep["all_scalarized_0.8_0.2"]["region_coverage_mean"]
    nehvi_cov = sweep["all_nehvi"]["region_coverage_mean"]
    parego_cov = sweep["all_parego"]["region_coverage_mean"]
    scal_spread = sweep["all_scalarized_0.8_0.2"]["angular_spread_mean"]
    nehvi_spread = sweep["all_nehvi"]["angular_spread_mean"]
    parego_spread = sweep["all_parego"]["angular_spread_mean"]

    # 4. fixed scalarization touches fewer objective-space regions than nehvi.
    crit.append(
        Criterion(
            "4. fixed scalarization concentrates",
            scal_cov < nehvi_cov,
            f"region coverage scalarized_0.8_0.2 {scal_cov:.2f} < nehvi {nehvi_cov:.2f} "
            f"(angular spread {scal_spread:.3f} vs {nehvi_spread:.3f})",
            hint="weights not steering: increase front concavity / rotation (Step 6)",
        )
    )

    # 5. ParEGO explores more trade-off regions than fixed scalarization.
    crit.append(
        Criterion(
            "5. ParEGO explores varied trade-offs",
            parego_cov > scal_cov,
            f"region coverage parego {parego_cov:.2f} > scalarized_0.8_0.2 {scal_cov:.2f} "
            f"(angular spread {parego_spread:.3f} vs {scal_spread:.3f})",
            hint="front too narrow for varied weights to matter: widen front regions (Step 6)",
        )
    )

    # 6. leaderboard not predetermined: >1 distinct per-seed winner OR a close top race.
    n_seeds = len(nehvi_by_seed)
    per_seed_winner = [
        max(sweep, key=lambda nm: sweep[nm]["auc_hv_by_seed"][i]) for i in range(n_seeds)
    ]
    distinct_winners = len(set(per_seed_winner))
    ranked = sorted((v["auc_hv_mean"] for v in sweep.values()), reverse=True)
    close_race = len(ranked) >= 2 and (ranked[0] - ranked[1]) / (abs(ranked[0]) or 1.0) < 0.15
    crit.append(
        Criterion(
            "6. leaderboard not predetermined",
            distinct_winners >= 2 or close_race,
            f"{distinct_winners} distinct per-seed winner(s) {set(per_seed_winner)}; "
            f"top-two mean gap {(ranked[0] - ranked[1]):.4f}",
            hint="one strategy dominates: rebalance difficulty (Step 6) or budget (config.N_ROUNDS)",
        )
    )

    crit.append(
        Criterion(
            "7. campaign runtime fits schedule",
            campaign_seconds < 60.0,
            f"a full {config.N_ROUNDS}-round campaign took {campaign_seconds:.1f}s (target < 60s)",
            hint="too slow: reduce RAW_SAMPLES / MC_SAMPLES (config) or pool size (Step 4)",
        )
    )

    crit.append(
        Criterion(
            "8. plots written",
            plots_written > 0,
            f"{plots_written} PNG(s) under outputs/preflight/",
            hint="plotting failed; check matplotlib backend",
        )
    )

    crit.append(
        Criterion(
            "discrete acq margin (q vs q+1)",
            margin["abs_margin"] > 1e-3,
            f"qth {margin['qth_value']:.4f} vs next {margin['next_value']:.4f} "
            f"(abs {margin['abs_margin']:.4g}, rel {margin['rel_margin']:.4g})",
            hint="tight margin risks cross-machine drift; the discrete path stays robust but inspect Step 6",
        )
    )

    crit.append(
        Criterion(
            "9. achieved-vs-true front contrast (reveal)",
            True,
            f"all_nehvi final HV {reveal['achieved_hv']:.4f} vs true-front HV {reveal['true_hv']:.4f} "
            f"({100 * reveal['achieved_hv'] / reveal['true_hv']:.0f}% of max)",
            info=True,
        )
    )

    return crit


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def write_plots(
    strategies: dict[str, list[dict]],
    pool: VHSequencePool,
    oracle: AntibodyOracle,
    initial_ids: list[int],
    out_dir: Path,
    seed: int = config.SEED,
) -> int:
    """Write HV-curve and selection-coverage PNGs; return the count written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    # 1) HV curves, one line per strategy (single representative seed).
    fig, ax = plt.subplots(figsize=(7, 5))
    coverage = {}
    for name, rounds in strategies.items():
        result = run_campaign(rounds, pool, oracle, initial_ids, seed=seed)
        ax.plot(range(len(result["hv_history"])), result["hv_history"], marker="o", label=name)
        coverage[name] = result["selected_Y"]
    ax.set_xlabel("round")
    ax.set_ylabel("hypervolume")
    ax.set_title("Hypervolume vs round, by strategy")
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(out_dir / "hv_curves.png", dpi=120)
    plt.close(fig)
    written += 1

    # 2) Objective-space coverage scatter for three contrasting strategies.
    contrast = [n for n in ("all_nehvi", "all_scalarized_0.8_0.2", "all_parego") if n in coverage]
    fig, axes = plt.subplots(1, len(contrast), figsize=(5 * len(contrast), 5), squeeze=False)
    for ax, name in zip(axes[0], contrast):
        plotting.plot_objective_space(coverage[name], ref_point=config.REF_POINT, ax=ax, title=name)
    fig.tight_layout()
    fig.savefig(out_dir / "selection_coverage.png", dpi=120)
    plt.close(fig)
    written += 1

    return written


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _print_banner(crit: list[Criterion]) -> bool:
    print("\n" + "=" * 72)
    print("PREFLIGHT — outline §18.4 acceptance criteria")
    print("=" * 72)
    all_pass = True
    for c in crit:
        if c.info:
            tag = "INFO"
        else:
            tag = "PASS" if c.passed else "FAIL"
            all_pass = all_pass and c.passed
        print(f"[{tag}] {c.name}")
        print(f"       {c.detail}")
        if not c.passed and not c.info:
            print(f"       hint: {c.hint}")
    print("=" * 72)
    print("RESULT:", "ALL CRITERIA PASS — ready to freeze (Step 11)" if all_pass else "FAIL — do not freeze")
    print("=" * 72 + "\n")
    return all_pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Instructor preflight difficulty sweep")
    parser.add_argument("--seeds", type=int, default=3, help="number of campaign seeds to average")
    parser.add_argument("--n-rounds", type=int, default=config.N_ROUNDS)
    parser.add_argument("--out", default=str(config.OUTPUTS_DIR / "preflight"))
    args = parser.parse_args(argv)

    pool = VHSequencePool.from_files()
    oracle = AntibodyOracle.from_files(allow_true=True)
    initial_ids = data_initial_ids()
    strategies = default_strategies(args.n_rounds)
    seeds = [config.SEED + i for i in range(args.seeds)]

    print(f"Sweeping {len(strategies)} strategies x {len(seeds)} seeds "
          f"x {args.n_rounds} rounds on a {len(pool)}-sequence pool ...")
    sweep = run_sweep(strategies, pool, oracle, initial_ids, seeds)

    # Determinism: identical results on a repeat run.
    a = run_campaign(strategies["all_nehvi"], pool, oracle, initial_ids, seed=config.SEED)
    b = run_campaign(strategies["all_nehvi"], pool, oracle, initial_ids, seed=config.SEED)
    determinism_ok = a["selected_ids"] == b["selected_ids"] and a["hv_history"] == b["hv_history"]

    # Timing one full campaign.
    t0 = time.perf_counter()
    run_campaign(strategies["all_nehvi"], pool, oracle, initial_ids, seed=config.SEED)
    campaign_seconds = time.perf_counter() - t0

    margin = discrete_acq_margin(pool, oracle, initial_ids)

    # Reveal contrast (criterion 9): achieved vs true front HV.
    achieved = run_campaign(strategies["all_nehvi"], pool, oracle, initial_ids, seed=config.SEED)
    true_front = oracle.true_objectives
    reveal = {
        "achieved_hv": achieved["hv_history"][-1],
        "true_hv": metrics.compute_hypervolume(
            true_front[metrics.compute_pareto_mask(true_front)], config.REF_POINT
        ),
    }

    out_dir = Path(args.out)
    plots_written = write_plots(strategies, pool, oracle, initial_ids, out_dir)

    _print_summary_table(sweep)
    crit = evaluate_criteria(sweep, determinism_ok, margin, campaign_seconds, plots_written, reveal)
    all_pass = _print_banner(crit)
    print(f"artifacts: {out_dir}/hv_curves.png, {out_dir}/selection_coverage.png")
    return 0 if all_pass else 1


def data_initial_ids() -> list[int]:
    from mobo_lab import data

    return data.load_initial_ids()


def _print_summary_table(sweep: dict[str, dict]) -> None:
    print(f"\n{'strategy':<26}{'AUC-HV':>9}{'finalHV':>9}{'angSpr':>8}{'cover':>7}{'nondom':>8}")
    print("-" * 67)
    for name, m in sorted(sweep.items(), key=lambda kv: kv[1]["auc_hv_mean"], reverse=True):
        print(f"{name:<26}{m['auc_hv_mean']:>9.4f}{m['final_hv_mean']:>9.4f}"
              f"{m['angular_spread_mean']:>8.3f}{m['region_coverage_mean']:>7.1f}{m['n_nondom_mean']:>8.1f}")


if __name__ == "__main__":
    raise SystemExit(main())
