"""Offline tests for ``scripts/build_latents.py`` (end-to-end build + guards)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mobo_lab import config
from mobo_lab.data import LIBRARY_COLUMNS

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_latents as bl  # noqa: E402


def _distinct_sequences() -> list[str]:
    return [
        "ACDEFGHIKLMNPQRSTVWY",
        "AAAACCCCDDDDEEEEFFFF",
        "KLKLKLKLKLKLKLKLKLKL",
        "MNPQRSTVWYMNPQRSTVWY",
        "GGGGGGSSSSSSAAAAAAYY",
        "WYWYWYFWFWFWYHYHYHYH",
        "DEDEDEKRKRKRDEKRDEKR",
        "VVVVIIIILLLLMMMMFFFF",
    ]


def _write_library(tmp_path: Path, sequences: list[str]) -> Path:
    rows = [
        {
            "sequence_id": f"VH-{i + 1:03d}",
            "sequence": seq,
            "length": len(seq),
            "source": "u",
            "cluster_id": 0,
        }
        for i, seq in enumerate(sequences)
    ]
    path = tmp_path / "vh_library.csv"
    pd.DataFrame(rows, columns=LIBRARY_COLUMNS).to_csv(path, index=False)
    return path


def test_main_writes_valid_npy(tmp_path):
    lib = _write_library(tmp_path, _distinct_sequences())
    out = tmp_path / "vh_latents.npy"
    bl.main(["--library", str(lib), "--out", str(out)])

    arr = np.load(out)
    assert arr.shape == (8, config.LATENT_DIM)
    assert arr.min() >= 0.0
    assert arr.max() <= 1.0


def test_check_latents_raises_on_collapsed_pair():
    # rows 0 and 1 are identical -> distance 0 -> below tau
    latents = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    with pytest.raises(ValueError, match="too close"):
        bl.check_latents(latents, ["VH-001", "VH-002", "VH-003"])


def test_check_latents_accepts_well_separated():
    latents = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [0.5, 0.0, 0.5, 0.0, 0.5],
        ]
    )
    dist = bl.check_latents(latents, ["VH-001", "VH-002", "VH-003"])
    assert dist > bl.MIN_PAIRWISE_TAU


def test_main_raises_on_duplicate_sequence(tmp_path):
    seqs = _distinct_sequences()
    seqs[1] = seqs[0]  # duplicate sequence (distinct IDs) -> identical latents
    lib = _write_library(tmp_path, seqs)
    out = tmp_path / "vh_latents.npy"
    with pytest.raises(ValueError, match="too close"):
        bl.main(["--library", str(lib), "--out", str(out)])
