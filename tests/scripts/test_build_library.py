"""Offline tests for ``scripts/build_library.py`` curation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from mobo_lab.data import LIBRARY_COLUMNS

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_library as bl  # noqa: E402


def test_composition_features_known_counts():
    feats = bl.composition_features(["AACC"])  # 2 A, 2 C, length 4
    assert feats[0, bl.AA_ORDER.index("A")] == 0.5
    assert feats[0, bl.AA_ORDER.index("C")] == 0.5
    assert feats[0, -1] == 4  # length feature


def test_keep_after_dropping_small():
    labels = np.array([0, 0, 0, 1, 1, 2])  # cluster sizes 3, 2, 1
    mask = bl.keep_after_dropping_small(labels, min_cluster_size=3)
    assert mask.tolist() == [True, True, True, False, False, False]


def test_balanced_sample_round_robin():
    labels = np.array([0, 0, 0, 1, 1, 1])
    idx = np.arange(6)
    picked = bl.balanced_sample(labels, idx, size=4, seed=0)
    assert len(picked) == 4
    # round-robin across two equal clusters -> 2 from each
    from_cluster0 = sum(1 for p in picked if labels[p] == 0)
    assert from_cluster0 == 2


def _grouped_df():
    """Three tight composition clusters (A/K/D rich) plus one isolated W-rich singleton."""
    rows = []
    base_len = 100
    for dominant, n in [("A", 10), ("K", 10), ("D", 10), ("W", 1)]:
        for i in range(n):
            seq = list(dominant * base_len)
            seq[i] = "C"  # make each sequence distinct
            rows.append({"sequence": "".join(seq), "source": f"{dominant}_unit", "isotype": "IGHG"})
    return pd.DataFrame(rows)


def test_curate_schema_unique_and_deterministic():
    df = _grouped_df()
    out1 = bl.curate(df, size=15, seed=0, n_clusters=4, min_cluster_size=3)
    out2 = bl.curate(df, size=15, seed=0, n_clusters=4, min_cluster_size=3)

    assert list(out1.columns) == LIBRARY_COLUMNS
    assert out1["sequence_id"].is_unique
    assert len(out1) <= 15
    pd.testing.assert_frame_equal(out1, out2)  # deterministic


def test_curate_drops_isolated_mode():
    df = _grouped_df()
    w_seq = df[df["source"] == "W_unit"]["sequence"].iloc[0]
    out = bl.curate(df, size=20, seed=0, n_clusters=4, min_cluster_size=3)
    # the lone W-rich sequence is a singleton cluster and must be dropped
    assert w_seq not in set(out["sequence"])


def test_curate_exact_dedup():
    df = pd.DataFrame(
        {
            "sequence": ["AAAA", "AAAA", "CCCC"],
            "source": ["u", "u", "u"],
            "isotype": ["IGHG"] * 3,
        }
    )
    out = bl.curate(df, size=10, seed=0, n_clusters=2, min_cluster_size=1)
    assert sorted(out["sequence"]) == ["AAAA", "CCCC"]
