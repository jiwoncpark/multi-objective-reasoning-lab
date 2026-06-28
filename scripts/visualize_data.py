"""Visualize the antibody VH library: biophysical properties, sequences, latents.

These plots are an instructor-only design aid. We use them to understand the
*shape* of the data -- the simulated biophysical measurements (ranges,
correlations, trade-offs) and the geometry of the latent design space -- so that
the synthetic oracle and the nearest-neighbour projection behave sensibly. The
students never see this script; they only interact with the oracle and the
notebooks.

Three groups of figures are produced (saved to ``docs/figures/`` by default).

Legacy biophysical EDA (113-sequence ``vh_data.csv`` reference; skipped if the
spreadsheet is absent):

1. ``property_distributions.png`` -- histograms of every numeric property.
2. ``property_correlations.png``  -- correlation heatmap between properties.
3. ``property_tradeoffs.png``     -- the two oracle-relevant trade-off views,
   with each raw property re-expressed as "higher is better."
4. ``sequence_overview.png``      -- sequence length, germline families, and
   amino-acid composition.

Latent-geometry EDA (the 2048-sequence curated pool, ``data/vh_latents.npy``;
skipped if absent):

5. ``latent_distribution.png``    -- 5x5 corner plot of the latent cube
   ``[0, 1]^5``: per-dimension marginals on the diagonal, pairwise 2D-density
   panels below, so the joint distribution of the 2048 sequences is visible.
6. ``latent_nn_distances.png``    -- histogram of each sequence's nearest-
   neighbour distance, with the global minimum marked (this is the quantity the
   projection-separation guard in ``build_latents.py`` protects).

Oracle objective ("property") space EDA (the hidden ``data/oracle_true_objectives.npy``;
skipped if absent):

7. ``objective_space.png``        -- the two synthetic objectives ("binding-like"
   vs "stability-like") for all 2048 sequences, with marginal histograms, the true
   Pareto front (staircase), the shared initial design, and the reference point.
   This is the instructor view of what teams start from and what they are chasing.

Notes on direction-of-goodness for the four simulated properties
----------------------------------------------------------------
* ``tm_celsius``       higher is better  (thermostability / developability)
* ``yield_mg_per_10ml`` higher is better (expression / developability)
* ``bv_elisa_score``    lower  is better (baculovirus polyspecificity; off-target)
* ``affinity_kd_nm``    lower  is better (dissociation constant; tighter binding)

To make trade-offs visually intuitive, the trade-off figure converts affinity to
``pKd = -log10(Kd in molar)`` (higher = tighter binding) and flips the BV score,
so every axis points the "good" way.

Usage
-----
    python scripts/visualize_data.py
    python scripts/visualize_data.py --csv data/vh_data.csv --outdir docs/figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Allow importing the sibling parse module so we can regenerate the table if the
# CSV is missing. ``scripts/`` is not an installed package, so we add it to path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_data import DEFAULT_OUTPUT, STANDARD_AMINO_ACIDS, parse_vh_data  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from mobo_lab import config  # noqa: E402

DEFAULT_OUTDIR = REPO_ROOT / "docs" / "figures"

# Below this nearest-neighbour distance two latent vectors are effectively
# indistinguishable to the projection step; mirrors the guard in build_latents.py.
LATENT_MIN_DIST_TAU = 1e-2

# The four simulated biophysical objectives, with a human-readable label and the
# direction that counts as "better." Driven off this table so the plots and the
# oracle-design discussion stay consistent.
PROPERTIES = {
    "tm_celsius": ("Tm (deg C)", "higher"),
    "bv_elisa_score": ("BV ELISA score", "lower"),
    "affinity_kd_nm": ("Affinity Kd (nM)", "lower"),
    "yield_mg_per_10ml": ("Yield (mg / 10 mL)", "higher"),
}

sns.set_theme(style="whitegrid", context="talk")


def load_table(csv_path: Path) -> pd.DataFrame:
    """Load the cleaned table, regenerating it from the .xlsx if absent."""
    if csv_path.exists():
        return pd.read_csv(csv_path)
    print(f"[visualize] {csv_path} not found; parsing the spreadsheet directly.")
    return parse_vh_data()


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'higher-is-better' versions of the binding/polyspecificity columns.

    * ``pkd``        = -log10(Kd in molar). Kd is given in nM, so Kd[M] = Kd_nM * 1e-9.
      Larger pKd  ==> tighter binding.
    * ``neg_bv``     = -bv_elisa_score. Larger ==> less polyspecific.
    """
    df = df.copy()
    df["pkd"] = -np.log10(df["affinity_kd_nm"] * 1e-9)
    df["neg_bv"] = -df["bv_elisa_score"]
    return df


def plot_property_distributions(df: pd.DataFrame, outdir: Path) -> Path:
    """Histograms of length, germline identity, and the four properties."""
    cols = [
        ("length", "Sequence length (aa)", False),
        ("germline_identity_pct", "Germline identity (%)", False),
        ("tm_celsius", "Tm (deg C)  [higher better]", False),
        ("yield_mg_per_10ml", "Yield (mg/10mL)  [higher better]", False),
        ("bv_elisa_score", "BV ELISA score  [lower better]", False),
        ("affinity_kd_nm", "Affinity Kd (nM)  [lower better, log x]", True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (col, label, log_x) in zip(axes.ravel(), cols):
        data = df[col]
        if log_x:
            bins = np.logspace(np.log10(data.min()), np.log10(data.max()), 25)
            ax.hist(data, bins=bins, color="#4C72B0", edgecolor="white")
            ax.set_xscale("log")
        else:
            ax.hist(data, bins=20, color="#4C72B0", edgecolor="white")
        ax.axvline(data.median(), color="#C44E52", linestyle="--", linewidth=2)
        ax.set_xlabel(label, fontsize=13)
        ax.set_ylabel("count", fontsize=13)
        ax.tick_params(labelsize=11)
    fig.suptitle(
        "VH library: property distributions (red dashed = median)", fontsize=18
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = outdir / "property_distributions.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_property_correlations(df: pd.DataFrame, outdir: Path) -> Path:
    """Correlation heatmap among the 'higher-is-better' property axes.

    Using the direction-normalised axes (pKd, Tm, yield, -BV) means a positive
    correlation always reads as "these two good things go together," which is
    what we care about when reasoning about whether objectives genuinely trade
    off (the premise of the whole lab).
    """
    cols = {
        "pkd": "pKd (binding)",
        "tm_celsius": "Tm (stability)",
        "yield_mg_per_10ml": "Yield",
        "neg_bv": "-BV (specificity)",
        "germline_identity_pct": "Germline id %",
    }
    corr = df[list(cols)].rename(columns=cols).corr(method="spearman")
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(
        corr,
        annot=True,
        fmt="+.2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Spearman rho"},
        ax=ax,
    )
    ax.set_title(
        "Property correlations (all axes oriented 'higher = better')",
        fontsize=15,
    )
    fig.tight_layout()
    path = outdir / "property_correlations.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _pareto_front_2d_max(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Boolean mask of the non-dominated points for 2-objective *maximization*.

    A point is non-dominated if no other point is >= it on both axes and strictly
    greater on at least one. O(n^2) is plenty for ~100 antibodies.
    """
    n = len(x)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        # i is dominated if some j is at least as good on both, better on one.
        better_eq = (x >= x[i]) & (y >= y[i])
        strictly = (x > x[i]) | (y > y[i])
        if np.any(better_eq & strictly & (np.arange(n) != i)):
            dominated[i] = True
    return ~dominated


def _pareto_staircase(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Staircase coordinates tracing the upper-right (max/max) Pareto frontier.

    Given the non-dominated points, sort by x ascending (y then descends) and
    connect them with horizontal-then-vertical steps, so the line bounds the
    dominated region the way a Pareto front is usually drawn.
    """
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    step_x, step_y = [xs[0]], [ys[0]]
    for k in range(1, len(xs)):
        step_x.extend([xs[k], xs[k]])   # move right, then down
        step_y.extend([ys[k - 1], ys[k]])
    return np.asarray(step_x), np.asarray(step_y)


def plot_property_tradeoffs(df: pd.DataFrame, outdir: Path) -> Path:
    """Corner plot of all property pairs, with the 2D Pareto front in each panel.

    Every axis is oriented "higher = better" (binding pKd, stability Tm, yield,
    specificity -BV), so in each off-diagonal panel the desirable corner is the
    top-right and the highlighted points + staircase show the achievable Pareto
    front for that pair. This lets us eyeball, for *every* pair of properties,
    whether the trade-off is genuine (a sloped front) or degenerate (a tight
    diagonal cloud) before committing the oracle to a particular objective pair.

    Layout: a lower-triangular corner plot -- diagonal holds each property's
    histogram; cell (row i, col j<i) scatters property j (x) vs property i (y).
    """
    # The four objective axes, all already "higher is better" in `df`.
    axes_def = [
        ("pkd", "pKd\n(binding)"),
        ("tm_celsius", "Tm\n(stability)"),
        ("yield_mg_per_10ml", "Yield"),
        ("neg_bv", "-BV\n(specificity)"),
    ]
    keys = [k for k, _ in axes_def]
    labels = [lab for _, lab in axes_def]
    n = len(keys)

    fig, axgrid = plt.subplots(n, n, figsize=(15, 14))

    for i in range(n):
        for j in range(n):
            ax = axgrid[i, j]

            if j > i:  # upper triangle: unused
                ax.axis("off")
                continue

            if i == j:  # diagonal: marginal histogram
                ax.hist(df[keys[i]], bins=18, color="#4C72B0", edgecolor="white")
                ax.set_yticks([])
            else:  # lower triangle: scatter j (x) vs i (y) + Pareto front
                x = df[keys[j]].to_numpy()
                y = df[keys[i]].to_numpy()
                ax.scatter(x, y, s=22, color="#B0B0B0", edgecolor="none", alpha=0.7)

                front = _pareto_front_2d_max(x, y)
                step_x, step_y = _pareto_staircase(x[front], y[front])
                ax.plot(step_x, step_y, color="#C44E52", linewidth=1.5, zorder=2)
                ax.scatter(
                    x[front], y[front], s=42, color="#C44E52",
                    edgecolor="k", linewidth=0.4, zorder=3,
                )

            # Labels only on the outer edges to keep the grid readable.
            if i == n - 1:
                ax.set_xlabel(labels[j], fontsize=12)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(labels[i], fontsize=12)
            elif i != j:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=9)

    fig.suptitle(
        "Property corner plot -- every axis higher=better; red = 2D Pareto front\n"
        "(top-right corner is desirable; sloped front = genuine trade-off)",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = outdir / "property_tradeoffs.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_sequence_overview(df: pd.DataFrame, outdir: Path, top_n: int = 12) -> Path:
    """Sequence-level view: length, top germline families, AA composition."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # (a) length distribution
    axes[0].hist(df["length"], bins=range(df["length"].min(), df["length"].max() + 2),
                 color="#55A868", edgecolor="white", align="left")
    axes[0].set_xlabel("Sequence length (aa)")
    axes[0].set_ylabel("count")
    axes[0].set_title("Length distribution")

    # (b) top germline families
    top = df["germline"].value_counts().head(top_n).iloc[::-1]
    axes[1].barh(top.index, top.values, color="#8172B3", edgecolor="white")
    axes[1].set_xlabel("count")
    axes[1].set_title(f"Top {top_n} germline genes (of {df['germline'].nunique()})")
    axes[1].tick_params(axis="y", labelsize=11)

    # (c) amino-acid composition across all sequences
    all_aa = "".join(df["sequence"])
    counts = pd.Series({aa: all_aa.count(aa) for aa in sorted(STANDARD_AMINO_ACIDS)})
    freq = counts / counts.sum() * 100
    axes[2].bar(freq.index, freq.values, color="#C44E52", edgecolor="white")
    axes[2].set_xlabel("amino acid")
    axes[2].set_ylabel("frequency (%)")
    axes[2].set_title("Amino-acid composition")

    fig.suptitle("VH library: sequence overview", fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = outdir / "sequence_overview.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Latent-geometry EDA (the curated 2048-sequence pool in [0, 1]^LATENT_DIM)
# --------------------------------------------------------------------------- #
def nearest_neighbor_distances(points: np.ndarray) -> np.ndarray:
    """Per-row Euclidean distance to the closest other row.

    Returns an array of length ``len(points)`` (zeros if fewer than two points).
    Uses the ``||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b`` identity, so the full
    pairwise sweep over the ~2k-point pool stays a couple of cheap matrix ops.
    """
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n < 2:
        return np.zeros(n)
    gram = points @ points.T
    sq = np.diag(gram)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * gram, 0.0)
    np.fill_diagonal(d2, np.inf)
    return np.sqrt(d2.min(axis=1))


def plot_latent_distribution(latents: np.ndarray, outdir: Path) -> Path:
    """Corner plot of the latent cube: marginals on the diagonal, density below.

    Diagonal cell ``i`` is the 1-D histogram of latent dimension ``i``; lower-
    triangle cell ``(i, j<i)`` is a 2-D density (``hist2d``) of dimension ``j`` (x)
    vs dimension ``i`` (y). All axes are pinned to ``[0, 1]`` so the panels share
    a common frame and a clumped or edge-piled distribution is obvious at a glance.
    """
    d = latents.shape[1]
    fig, axgrid = plt.subplots(d, d, figsize=(3.0 * d, 2.8 * d))
    for i in range(d):
        for j in range(d):
            ax = axgrid[i, j]
            if j > i:  # upper triangle: unused
                ax.axis("off")
                continue
            if i == j:  # diagonal: marginal histogram
                ax.hist(latents[:, i], bins=30, range=(0, 1),
                        color="#4C72B0", edgecolor="white")
                ax.set_yticks([])
                ax.set_xlim(0, 1)
            else:  # lower triangle: 2-D density
                ax.hist2d(latents[:, j], latents[:, i], bins=30,
                          range=[[0, 1], [0, 1]], cmap="mako")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

            if i == d - 1:
                ax.set_xlabel(f"z{j}", fontsize=12)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(f"z{i}", fontsize=12)
            elif i != j:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=9)

    fig.suptitle(
        f"Latent distribution of {len(latents)} sequences in [0, 1]^{d}\n"
        "(diagonal = per-dimension marginal; lower = pairwise 2D density)",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = outdir / "latent_distribution.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_latent_nn_distances(
    latents: np.ndarray, outdir: Path, tau: float = LATENT_MIN_DIST_TAU
) -> Path:
    """Histogram of nearest-neighbour distances, with the global minimum marked."""
    nn = nearest_neighbor_distances(latents)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(nn, bins=40, color="#55A868", edgecolor="white")
    ax.axvline(nn.min(), color="#C44E52", linestyle="--", linewidth=2,
               label=f"min = {nn.min():.4f}")
    ax.axvline(np.median(nn), color="#4C72B0", linestyle=":", linewidth=2,
               label=f"median = {np.median(nn):.4f}")
    ax.axvline(tau, color="#8172B3", linestyle="-", linewidth=1.5,
               label=f"guard tau = {tau:g}")
    ax.set_xlabel("nearest-neighbour distance in latent space")
    ax.set_ylabel("count")
    ax.set_title(
        f"Latent separation of {len(latents)} sequences "
        "(left tail near tau = projection risk)"
    )
    ax.legend()
    fig.tight_layout()
    path = outdir / "latent_nn_distances.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Oracle objective ("property") space EDA (the two synthetic objectives)
# --------------------------------------------------------------------------- #
def plot_objective_space(
    Y: np.ndarray,
    outdir: Path,
    initial_ids: list[int] | None = None,
    ref_point=None,
) -> Path:
    """Joint plot of the two oracle objectives with the true Pareto front.

    Central panel scatters all sequences in (objective 1, objective 2) space (both
    maximization), with the true Pareto front drawn as a staircase, the shared
    initial design circled, and the reference point starred. Marginal histograms
    sit above and to the right. This is the instructor's view of the design problem:
    where teams start and which trade-offs they are competing to discover.
    """
    import torch  # local: keep the default biophysical path import-light

    from mobo_lab.metrics import compute_pareto_mask

    mask = compute_pareto_mask(torch.as_tensor(Y, dtype=torch.double)).numpy()
    rho = pd.Series(Y[:, 0]).corr(pd.Series(Y[:, 1]), method="spearman")

    fig = plt.figure(figsize=(10, 9))
    gs = fig.add_gridspec(
        2, 2, width_ratios=(4, 1), height_ratios=(1, 4), wspace=0.04, hspace=0.04
    )
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)

    # all library sequences
    ax.scatter(Y[:, 0], Y[:, 1], s=10, color="#B0B0B0", alpha=0.55,
               edgecolor="none", label=f"library ({len(Y)})")

    # initial design (the shared starter set)
    if initial_ids is not None and len(initial_ids):
        init = np.asarray(initial_ids, dtype=int)
        ax.scatter(Y[init, 0], Y[init, 1], s=80, facecolor="none",
                   edgecolor="#4C72B0", linewidth=1.8, zorder=4,
                   label=f"initial design ({len(init)})")

    # true Pareto front: staircase + points
    fx, fy = Y[mask, 0], Y[mask, 1]
    step_x, step_y = _pareto_staircase(fx, fy)
    ax.plot(step_x, step_y, color="#C44E52", linewidth=2, zorder=3)
    ax.scatter(fx, fy, s=55, color="#C44E52", edgecolor="k", linewidth=0.5,
               zorder=5, label=f"true Pareto front ({int(mask.sum())})")

    # reference point
    if ref_point is not None:
        ax.scatter([ref_point[0]], [ref_point[1]], marker="*", s=220, color="k",
                   zorder=6, label="reference point")
        ax.axvline(ref_point[0], color="k", linestyle=":", linewidth=1, alpha=0.4)
        ax.axhline(ref_point[1], color="k", linestyle=":", linewidth=1, alpha=0.4)

    ax.set_xlabel("objective 1: binding-like  (higher = better)", fontsize=12)
    ax.set_ylabel("objective 2: stability-like  (higher = better)", fontsize=12)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)

    # marginal histograms
    ax_top.hist(Y[:, 0], bins=40, color="#B0B0B0", edgecolor="white")
    ax_top.axis("off")
    ax_right.hist(Y[:, 1], bins=40, orientation="horizontal",
                  color="#B0B0B0", edgecolor="white")
    ax_right.axis("off")

    ax_top.set_title(
        f"Oracle objective ('property') space -- {len(Y)} sequences   "
        f"(Spearman rho = {rho:+.2f})",
        fontsize=14,
    )
    path = outdir / "objective_space.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Cleaned CSV from parse_data.py (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--latents",
        type=Path,
        default=config.LATENTS_NPY,
        help=f"Latent design matrix from build_latents.py (default: {config.LATENTS_NPY}).",
    )
    parser.add_argument(
        "--true",
        type=Path,
        default=config.ORACLE_TRUE_NPY,
        help=f"Hidden true objectives from build_oracle.py (default: {config.ORACLE_TRUE_NPY}).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help=f"Directory for output figures (default: {DEFAULT_OUTDIR}).",
    )
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    # (1) Legacy biophysical EDA over the 113-sequence reference table.
    df = add_derived_columns(load_table(args.csv))
    written += [
        plot_property_distributions(df, args.outdir),
        plot_property_correlations(df, args.outdir),
        plot_property_tradeoffs(df, args.outdir),
        plot_sequence_overview(df, args.outdir),
    ]
    print(f"Visualized {len(df)} reference sequences (biophysical EDA).")

    # (2) Latent-geometry EDA over the curated competition pool.
    if args.latents.exists():
        latents = np.load(args.latents)
        written += [
            plot_latent_distribution(latents, args.outdir),
            plot_latent_nn_distances(latents, args.outdir),
        ]
        print(f"Visualized {len(latents)} latent vectors in [0, 1]^{latents.shape[1]}.")
    else:
        print(f"[visualize] {args.latents} not found; skipping latent figures "
              "(run scripts/build_latents.py first).")

    # (3) Oracle objective ("property") space over the curated competition pool.
    if args.true.exists():
        Y_true = np.load(args.true)
        initial_ids = None
        if config.INITIAL_IDS_JSON.exists():
            from mobo_lab import data as _data

            initial_ids = _data.load_initial_ids(config.INITIAL_IDS_JSON)
        written.append(
            plot_objective_space(Y_true, args.outdir, initial_ids, config.REF_POINT)
        )
        print(f"Visualized {len(Y_true)} sequences in oracle objective space.")
    else:
        print(f"[visualize] {args.true} not found; skipping objective-space figure "
              "(run scripts/build_oracle.py first).")

    print(f"Wrote {len(written)} figures:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
