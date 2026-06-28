"""Choose the fixed initial design ``data/initial_indices.json`` shared by all teams.

Every team starts a campaign from the **same** ``N_INITIAL`` evaluated sequences.
We want that starter set to be:

* **diverse in latent space**, so the initial surrogate has global coverage to fit
  -- chosen by farthest-point sampling over ``data/vh_latents.npy``; and
* **non-front-saturating**, so the true Pareto front is left to be discovered
  during the campaign (outline section 18.3.4) -- we sample only from sequences that
  are *not* on the true front, so the initial design seeds zero Pareto-optimal
  points.

Output is ``{"seed": SEED, "initial_ids": [...]}`` where ids are row indices into
``data/vh_library.csv``. The procedure is fully deterministic.

Usage
-----
    python scripts/build_initial_design.py
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

DEFAULT_OUT = config.INITIAL_IDS_JSON


def farthest_point_sampling(
    X: np.ndarray, k: int, candidate_idx, start_idx: int
) -> list[int]:
    """Greedy max-min (farthest-point) sampling of ``k`` rows from ``candidate_idx``.

    Starting from ``start_idx``, repeatedly add the candidate whose distance to the
    nearest already-selected point is largest. Deterministic (ties broken by lowest
    index, via ``argmax``).
    """
    X = np.asarray(X, dtype=float)
    cand = np.array([int(i) for i in candidate_idx if int(i) != int(start_idx)], dtype=int)
    selected = [int(start_idx)]
    if len(cand) == 0:
        return selected
    min_d = ((X[cand] - X[start_idx]) ** 2).sum(axis=1)  # sq-dist to nearest selected
    while len(selected) < k and len(cand) > 0:
        j = int(np.argmax(min_d))
        nxt = int(cand[j])
        selected.append(nxt)
        keep = np.arange(len(cand)) != j
        cand, min_d = cand[keep], min_d[keep]
        if len(cand) > 0:
            d_new = ((X[cand] - X[nxt]) ** 2).sum(axis=1)
            min_d = np.minimum(min_d, d_new)
    return selected


def choose_initial_ids(
    X: np.ndarray,
    Y_true: np.ndarray,
    n_initial: int = config.N_INITIAL,
    max_quantile: float = 0.6,
) -> list[int]:
    """Pick ``n_initial`` diverse, low-hypervolume starter indices (sorted, deterministic).

    To leave the competition plenty of headroom, candidates are restricted to the
    **central-or-below** region of objective space -- sequences whose *both*
    objectives sit at or below the ``max_quantile`` quantile of the library. This
    excludes the high-objective and trade-off-extreme points that would otherwise
    inflate the initial hypervolume. Within that band we still run farthest-point
    sampling so the starter set keeps good latent-space spread for the surrogate.
    If the band is too small for ``n_initial``, we relax to all non-front rows, then
    to all rows.
    """
    import torch

    from mobo_lab.metrics import compute_pareto_mask

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y_true, dtype=float)
    front = compute_pareto_mask(torch.as_tensor(Y, dtype=torch.double)).numpy()

    hi1 = np.quantile(Y[:, 0], max_quantile)
    hi2 = np.quantile(Y[:, 1], max_quantile)
    band = (~front) & (Y[:, 0] <= hi1) & (Y[:, 1] <= hi2)

    candidates = np.where(band)[0]
    if len(candidates) < n_initial:  # band too tight -> relax to all non-front rows
        candidates = np.where(~front)[0]
    if len(candidates) < n_initial:  # degenerate tiny input -> fall back to all rows
        candidates = np.arange(len(X))

    centroid = X.mean(axis=0)
    start = int(candidates[np.argmax(((X[candidates] - centroid) ** 2).sum(axis=1))])
    ids = farthest_point_sampling(X, n_initial, candidates, start)
    return sorted(ids)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Choose the shared initial design")
    parser.add_argument("--latents", default=str(config.LATENTS_NPY))
    parser.add_argument("--true", default=str(config.ORACLE_TRUE_NPY))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--n-initial", type=int, default=config.N_INITIAL)
    parser.add_argument("--max-quantile", type=float, default=0.6,
                        help="keep starters with both objectives at/below this quantile (headroom).")
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args(argv)

    X = data.load_latents(args.latents).numpy()
    Y_true = data.load_true_objectives(args.true).numpy()
    ids = choose_initial_ids(X, Y_true, n_initial=args.n_initial, max_quantile=args.max_quantile)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"seed": args.seed, "initial_ids": ids}, indent=2))

    # Report front saturation and the headroom this starter set leaves.
    import torch

    from mobo_lab.metrics import compute_hypervolume, compute_pareto_mask

    Yt = torch.as_tensor(Y_true, dtype=torch.double)
    front = compute_pareto_mask(Yt).numpy()
    hv_init = compute_hypervolume(Yt[ids], config.REF_POINT)
    hv_front = compute_hypervolume(Yt[front], config.REF_POINT)
    print(f"initial design: {len(ids)} ids -> {out_path}")
    print(f"  ids: {ids}")
    print(f"  on true Pareto front: {int(front[ids].sum())} (target 0; never all)")
    print(f"  HV(initial) = {hv_init:.4f}  vs  HV(true front) = {hv_front:.4f}  "
          f"({100 * hv_init / hv_front:.1f}% of max; lower = more headroom)")


if __name__ == "__main__":
    main()
