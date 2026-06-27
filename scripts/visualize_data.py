"""Visualize the antibody VH library: biophysical properties and sequences.

These plots are an instructor-only design aid. We use them to understand the
*shape* of the simulated biophysical measurements -- their ranges, correlations,
and trade-offs -- so that the synthetic oracle we build for the lab produces a
realistic-looking, non-trivial Pareto front. The students never see this script;
they only interact with the oracle and the notebooks.

Four figures are produced (saved to ``docs/figures/`` by default):

1. ``property_distributions.png`` -- histograms of every numeric property.
2. ``property_correlations.png``  -- correlation heatmap between properties.
3. ``property_tradeoffs.png``     -- the two oracle-relevant trade-off views,
   with each raw property re-expressed as "higher is better."
4. ``sequence_overview.png``      -- sequence length, germline families, and
   amino-acid composition.

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
DEFAULT_OUTDIR = REPO_ROOT / "docs" / "figures"

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Cleaned CSV from parse_data.py (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help=f"Directory for output figures (default: {DEFAULT_OUTDIR}).",
    )
    args = parser.parse_args()

    df = add_derived_columns(load_table(args.csv))
    args.outdir.mkdir(parents=True, exist_ok=True)

    written = [
        plot_property_distributions(df, args.outdir),
        plot_property_correlations(df, args.outdir),
        plot_property_tradeoffs(df, args.outdir),
        plot_sequence_overview(df, args.outdir),
    ]
    print(f"Visualized {len(df)} sequences. Wrote {len(written)} figures:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
