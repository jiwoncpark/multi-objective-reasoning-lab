"""Curate the filtered OAS sequences into the final candidate library.

Input is ``data/oas_filtered.csv.gz`` (produced by ``scripts/download_oas.py``):
already row-filtered and exact-deduped clean VH sequences. This script removes the
remaining redundancy and isolated/rare modes so the pool is evenly covered, then
samples it down to ``config.LIBRARY_SIZE`` (= 2048).

Pipeline
--------
1. Exact dedup (belt-and-suspenders).
2. Cheap composition features (20 amino-acid fractions + length).
3. k-means clustering (numpy only, deterministic given the seed).
4. Drop singleton/tiny clusters -- the "isolated rare modes" we don't want.
5. Round-robin sample across the surviving clusters to balance density, up to the
   target size.
6. Assign stable IDs and write ``data/vh_library.csv``.

Everything is deterministic given ``--seed``: identical input -> identical output.

Usage
-----
    python scripts/build_library.py --filtered data/oas_filtered.csv.gz \
        --size 2048 --out data/vh_library.csv
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from mobo_lab import config  # noqa: E402
from mobo_lab.data import LIBRARY_COLUMNS  # noqa: E402

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
DEFAULT_FILTERED = REPO_ROOT / "data" / "oas_filtered.csv.gz"
DEFAULT_OUT = config.LIBRARY_CSV


# --------------------------------------------------------------------------- #
# Curation building blocks (each unit-testable)
# --------------------------------------------------------------------------- #
def composition_features(sequences: list[str]) -> np.ndarray:
    """Return an ``[n, 21]`` matrix: 20 amino-acid fractions + sequence length."""
    feats = np.zeros((len(sequences), len(AA_ORDER) + 1))
    for i, seq in enumerate(sequences):
        length = len(seq) or 1
        for j, aa in enumerate(AA_ORDER):
            feats[i, j] = seq.count(aa) / length
        feats[i, len(AA_ORDER)] = len(seq)
    return feats


def standardize(features: np.ndarray) -> np.ndarray:
    mu = features.mean(axis=0)
    sd = features.std(axis=0)
    sd[sd == 0] = 1.0
    return (features - mu) / sd


def kmeans(features: np.ndarray, k: int, seed: int, iters: int = 100) -> np.ndarray:
    """Deterministic numpy k-means (k-means++ init). Returns integer cluster labels."""
    rng = np.random.default_rng(seed)
    n = len(features)
    k = max(1, min(k, n))

    first = int(rng.integers(n))
    centers = [first]
    d2 = ((features - features[first]) ** 2).sum(axis=1)
    for _ in range(1, k):
        total = d2.sum()
        probs = d2 / total if total > 0 else np.full(n, 1.0 / n)
        nxt = int(rng.choice(n, p=probs))
        centers.append(nxt)
        d2 = np.minimum(d2, ((features - features[nxt]) ** 2).sum(axis=1))

    centroids = features[centers].copy()
    labels = np.full(n, -1, dtype=int)
    for step in range(iters):
        dists = (
            (features**2).sum(axis=1)[:, None]
            - 2 * features @ centroids.T
            + (centroids**2).sum(axis=1)[None, :]
        )
        new_labels = dists.argmin(axis=1)
        if step > 0 and np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            members = features[labels == j]
            if len(members):
                centroids[j] = members.mean(axis=0)
    return labels


def keep_after_dropping_small(labels: np.ndarray, min_cluster_size: int) -> np.ndarray:
    """Boolean mask dropping points whose cluster has fewer than ``min_cluster_size`` members."""
    labels = np.asarray(labels)
    unique, counts = np.unique(labels, return_counts=True)
    size_of = dict(zip(unique.tolist(), counts.tolist()))
    return np.array([size_of[int(lab)] >= min_cluster_size for lab in labels])


def balanced_sample(labels: np.ndarray, orig_indices: np.ndarray, size: int, seed: int) -> list[int]:
    """Round-robin sample original indices across clusters to balance density."""
    rng = np.random.default_rng(seed)
    by_cluster: dict[int, list[int]] = {}
    for lab, idx in zip(np.asarray(labels).tolist(), np.asarray(orig_indices).tolist()):
        by_cluster.setdefault(int(lab), []).append(int(idx))
    for lab in by_cluster:
        rng.shuffle(by_cluster[lab])

    clusters = sorted(by_cluster)
    selected: list[int] = []
    while len(selected) < size:
        drew = False
        for lab in clusters:
            if by_cluster[lab]:
                selected.append(by_cluster[lab].pop())
                drew = True
                if len(selected) >= size:
                    break
        if not drew:
            break
    return selected


def curate(
    df: pd.DataFrame,
    size: int = config.LIBRARY_SIZE,
    seed: int = config.SEED,
    n_clusters: int | None = None,
    min_cluster_size: int = 3,
) -> pd.DataFrame:
    """Dedup, cluster, drop rare modes, balance-sample, and assign IDs."""
    df = df.drop_duplicates("sequence").reset_index(drop=True)
    sequences = df["sequence"].astype(str).tolist()
    n = len(sequences)

    features = standardize(composition_features(sequences))
    k = n_clusters if n_clusters is not None else max(2, int(round(np.sqrt(n))))
    labels = kmeans(features, k, seed)

    keep_mask = keep_after_dropping_small(labels, min_cluster_size)
    kept_idx = np.where(keep_mask)[0]
    if len(kept_idx) == 0:  # tiny input: nothing survives the cutoff -> keep all
        kept_idx = np.arange(n)

    selected = balanced_sample(labels[kept_idx], kept_idx, size, seed)
    if len(selected) < size:
        warnings.warn(
            f"only {len(selected)} sequences after curation (< requested {size}); "
            "widen the OAS download (raise --target-sequences / --max-units)",
            stacklevel=2,
        )

    out = df.iloc[selected].reset_index(drop=True).copy()
    out["length"] = out["sequence"].str.len()
    out["cluster_id"] = labels[selected]
    if "source" not in out.columns:
        out["source"] = "unknown"
    out.insert(0, "sequence_id", [f"VH-{i + 1:05d}" for i in range(len(out))])
    return out[LIBRARY_COLUMNS]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Curate filtered OAS sequences into vh_library.csv")
    parser.add_argument("--filtered", default=str(DEFAULT_FILTERED))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--size", type=int, default=config.LIBRARY_SIZE)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--n-clusters", type=int, default=None)
    parser.add_argument("--min-cluster-size", type=int, default=3)
    args = parser.parse_args(argv)

    df = pd.read_csv(args.filtered, compression="infer")
    n_in = len(df)
    out = curate(
        df,
        size=args.size,
        seed=args.seed,
        n_clusters=args.n_clusters,
        min_cluster_size=args.min_cluster_size,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(
        f"curated {n_in} filtered -> {len(out)} library sequences "
        f"({out['cluster_id'].nunique()} clusters) -> {out_path}"
    )


if __name__ == "__main__":
    main()
