"""Build the hidden true-objective table ``data/oracle_true_objectives.npy``.

The two objectives ("binding-like", "stability-like", both maximization) are a
**smooth synthetic function of the 5-D latents** (``data/vh_latents.npy``), not
real biophysical labels. Building them synthetically lets us hit the competition
difficulty targets on purpose (outline section 18.2):

* **near-independent trade-off** -- two nearly-orthogonal linear directions, so the
  rank correlation of the two objectives over the library is small (the original
  EDA's genuine binding-vs-stability trade-off, re-expressed as a design target);
* **>=2 separated Pareto regions** -- two Gaussian "bonus" bumps placed far apart in
  latent space push out two distinct front lobes, so fixed scalarization can
  over-focus and exploration can pay off;
* **mild front concavity** -- a shallow Gaussian "valley" between the two bumps
  recesses the balanced middle, so Chebyshev scalarization beats a linear weighted
  sum (motivates the notebook-04 extension).

Each objective is min-max normalized over the library to roughly ``[0, 1]`` so
``REF_POINT = [-0.05, -0.05]`` sits just below the worst real values. The noise
that students see is added later, deterministically, by ``mobo_lab.oracle``.

Coupling: the latents fix this geometry, so re-running ``build_latents.py`` with a
different backend requires rebuilding the oracle (and the frozen golden values).

Usage
-----
    python scripts/build_oracle.py
    python scripts/build_oracle.py --params data/oracle_params.json   # sweep difficulty
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from mobo_lab import config, data  # noqa: E402

DEFAULT_OUT = config.ORACLE_TRUE_NPY
DEFAULT_PARAMS_OUT = config.DATA_DIR / "oracle_params.json"

# Tunable design constants. ``a1``/``a2`` are near-orthogonal linear directions
# (disjoint latent support -> near-zero rank correlation, since the PCA latents are
# already near-uncorrelated). Each bump adds ``height * exp(-||x-c||^2 / 2 w^2)``
# scaled into each objective by ``obj``. Bumps A/B build the two separated front
# lobes; the valley (negative, both objectives) recesses the balanced middle.
DEFAULT_PARAMS: dict = {
    "a1": [1.6, 0.0, 1.1, 0.0, 0.0],   # binding-like: driven by z0, z2
    "a2": [0.0, 1.6, 0.0, 1.1, 0.0],   # stability-like: driven by z1, z3
    "bumps": [
        # extreme-binding lobe (high z0,z2 / low z1,z3): boost objective 1
        {"center": [0.70, 0.30, 0.65, 0.30, 0.50], "width": 0.22, "height": 0.55, "obj": [1.0, 0.0]},
        # extreme-stability lobe (low z0,z2 / high z1,z3): boost objective 2
        {"center": [0.30, 0.70, 0.30, 0.65, 0.50], "width": 0.22, "height": 0.55, "obj": [0.0, 1.0]},
        # concavity valley in the balanced middle: penalize both objectives
        {"center": [0.50, 0.50, 0.48, 0.48, 0.50], "width": 0.26, "height": -0.35, "obj": [1.0, 1.0]},
    ],
}


def _bump(X: np.ndarray, center, width: float) -> np.ndarray:
    d2 = ((X - np.asarray(center, dtype=float)) ** 2).sum(axis=1)
    return np.exp(-d2 / (2.0 * width**2))


def synthetic_objectives_raw(X: np.ndarray, params: dict = DEFAULT_PARAMS) -> np.ndarray:
    """Raw (un-normalized) synthetic objectives ``[N, 2]`` from latents ``X`` ``[N, 5]``."""
    X = np.asarray(X, dtype=float)
    g1 = X @ np.asarray(params["a1"], dtype=float)
    g2 = X @ np.asarray(params["a2"], dtype=float)
    for bump in params["bumps"]:
        shape = bump["height"] * _bump(X, bump["center"], bump["width"])
        m1, m2 = bump["obj"]
        g1 = g1 + m1 * shape
        g2 = g2 + m2 * shape
    return np.stack([g1, g2], axis=1)


def minmax_columns(Y: np.ndarray) -> np.ndarray:
    """Min-max each objective into ``[0, 1]`` over the library."""
    lo = Y.min(axis=0)
    span = Y.max(axis=0) - lo
    span = np.where(span == 0.0, 1.0, span)
    return (Y - lo) / span


def synthetic_objectives(X: np.ndarray, params: dict = DEFAULT_PARAMS) -> np.ndarray:
    """Final true objectives ``[N, 2]`` in ``[0, 1]`` (raw objectives, then min-max)."""
    return minmax_columns(synthetic_objectives_raw(X, params))


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation via Pearson on ordinal ranks (no scipy dependency)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the synthetic oracle true objectives")
    parser.add_argument("--latents", default=str(config.LATENTS_NPY))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--params", default=None, help="optional params JSON (default: built-in)")
    parser.add_argument("--params-out", default=str(DEFAULT_PARAMS_OUT))
    args = parser.parse_args(argv)

    params = json.loads(Path(args.params).read_text()) if args.params else DEFAULT_PARAMS
    X = data.load_latents(args.latents).numpy()
    Y = synthetic_objectives(X, params)

    # Diagnostics against the difficulty targets (outline section 18.2).
    from mobo_lab.metrics import compute_pareto_mask  # local import: keep header light
    import torch

    rho = spearman_corr(Y[:, 0], Y[:, 1])
    front_mask = compute_pareto_mask(torch.as_tensor(Y, dtype=torch.double)).numpy()
    front_X = X[front_mask]
    max_front_dist = (
        float(np.sqrt(((front_X[:, None, :] - front_X[None, :, :]) ** 2).sum(-1)).max())
        if len(front_X) > 1
        else 0.0
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, Y.astype(np.float64))
    Path(args.params_out).write_text(json.dumps(params, indent=2))

    print(f"oracle true objectives: shape {Y.shape} -> {out_path}")
    print(f"  obj1 range [{Y[:, 0].min():.3f}, {Y[:, 0].max():.3f}], "
          f"obj2 range [{Y[:, 1].min():.3f}, {Y[:, 1].max():.3f}]")
    print(f"  spearman(obj1, obj2) = {rho:+.3f}  (target |rho| small)")
    print(f"  Pareto front: {int(front_mask.sum())} points, "
          f"max pairwise latent distance {max_front_dist:.3f}  (>=2 separated regions)")
    print(f"  params -> {args.params_out}")


if __name__ == "__main__":
    main()
