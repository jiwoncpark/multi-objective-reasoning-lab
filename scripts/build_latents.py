"""Build the latent design matrix ``data/vh_latents.npy`` from the curated library.

Each curated VH sequence becomes a continuous vector ``x in [0, 1]^LATENT_DIM``
that BoTorch optimizes over a standard box. The default backend is deterministic,
CPU-only and needs no downloads (see ``mobo_lab/embeddings.py``).

Backend coupling: switching ``--backend`` changes the latent *geometry*, so the
oracle (Step 6) and the frozen golden-path constants (Steps 10-11) MUST be
regenerated afterwards.

Usage
-----
    python scripts/build_latents.py --backend descriptor_pca --out data/vh_latents.npy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from mobo_lab import config, data, embeddings  # noqa: E402

DEFAULT_LIBRARY = config.LIBRARY_CSV
DEFAULT_OUT = config.LATENTS_NPY
# Minimum allowed pairwise Euclidean distance between any two latent vectors. Two
# sequences that collapse this close would be near-indistinguishable to the
# nearest-neighbour projection (outline section 18.2.6), so we fail loudly and let
# the instructor drop/curate one upstream in build_library.py.
MIN_PAIRWISE_TAU = 1e-2


def check_latents(
    latents: np.ndarray,
    sequence_ids: list[str],
    tau: float = MIN_PAIRWISE_TAU,
) -> float:
    """Validate shape, range and separation; raise ``ValueError`` on any failure.

    Returns the minimum pairwise distance on success.
    """
    if latents.ndim != 2 or latents.shape[1] != config.LATENT_DIM:
        raise ValueError(
            f"latents must have shape [N, {config.LATENT_DIM}], got {tuple(latents.shape)}"
        )
    lo, hi = float(latents.min()), float(latents.max())
    if lo < 0.0 or hi > 1.0:
        raise ValueError(f"latents must lie in [0, 1], got range [{lo:.4g}, {hi:.4g}]")

    dist, i, j = embeddings.min_pairwise_distance(latents)
    if dist <= tau:
        id_i = sequence_ids[i] if 0 <= i < len(sequence_ids) else i
        id_j = sequence_ids[j] if 0 <= j < len(sequence_ids) else j
        raise ValueError(
            f"latents too close: {id_i} and {id_j} are {dist:.4g} apart (<= tau={tau:g}); "
            "drop or re-curate one of them upstream in build_library.py"
        )
    return dist


def summarize(latents: np.ndarray, info: dict, min_dist: float) -> None:
    print(f"latents: shape {tuple(latents.shape)}")
    print("per-dimension [min, max] (spread):")
    for d in range(latents.shape[1]):
        col = latents[:, d]
        print(f"  z{d}: [{col.min():.3f}, {col.max():.3f}]  (spread {col.max() - col.min():.3f})")
    evr = np.asarray(info["explained_variance_ratio"])
    print(
        "explained-variance ratio: "
        + ", ".join(f"{r:.3f}" for r in evr)
        + f"  (sum {evr.sum():.3f})"
    )
    print(f"min pairwise distance: {min_dist:.4g}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build vh_latents.npy from the curated library")
    parser.add_argument("--library", default=str(DEFAULT_LIBRARY))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--backend", default="descriptor_pca", choices=sorted(embeddings.BACKENDS))
    parser.add_argument("--tau", type=float, default=MIN_PAIRWISE_TAU)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args(argv)

    df = data.load_library(args.library)
    sequences = df["sequence"].astype(str).tolist()
    sequence_ids = df["sequence_id"].astype(str).tolist()

    latents, info = embeddings.build_latents_with_info(
        sequences, backend=args.backend, n_components=config.LATENT_DIM, seed=args.seed
    )
    min_dist = check_latents(latents, sequence_ids, tau=args.tau)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, latents)

    summarize(latents, info, min_dist)
    print(f"wrote {len(latents)} latents [{args.backend}] -> {out_path}")


if __name__ == "__main__":
    main()
