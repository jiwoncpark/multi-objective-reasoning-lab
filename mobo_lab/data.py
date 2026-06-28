"""Loaders for the curated candidate library and its derived data assets.

These are the functions the package and notebooks use to read the data files
produced by the instructor scripts (``scripts/download_oas.py``,
``scripts/build_library.py``, and the Step 5/6 embedding + oracle builders). Each
loader validates the contract the rest of the lab relies on, so a malformed or
stale file fails loudly here rather than deep inside a Bayesian-optimization loop.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import config

# The 20 standard amino-acid one-letter codes -- the canonical definition for the
# whole package. A clean VH sequence is a string drawn only from these (no gaps,
# stop codons, or ambiguous ``X``).
STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")

# Required columns of ``data/vh_library.csv`` (see docs/05).
LIBRARY_COLUMNS: list[str] = ["sequence_id", "sequence", "length", "source", "cluster_id"]


def _nonstandard_residues(sequence: str) -> set[str]:
    return set(sequence) - STANDARD_AMINO_ACIDS


def load_library(path: str | Path = config.LIBRARY_CSV) -> pd.DataFrame:
    """Load and validate the curated candidate library CSV.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    KeyError
        If a required column is missing.
    ValueError
        If ``sequence_id`` values are not unique or a sequence contains a
        non-standard amino-acid character.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"library CSV not found: {path}")
    df = pd.read_csv(path)

    missing = [c for c in LIBRARY_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"library CSV missing columns: {missing}")

    if df["sequence_id"].duplicated().any():
        raise ValueError("library CSV has duplicate sequence_id values")

    for seq in df["sequence"].astype(str):
        bad = _nonstandard_residues(seq)
        if bad:
            raise ValueError(f"sequence contains non-standard residues {sorted(bad)}: {seq[:20]}...")

    return df.reset_index(drop=True)


def load_sequences(path: str | Path = config.LIBRARY_CSV) -> list[str]:
    """Return just the amino-acid sequences from the curated library, in row order."""
    return load_library(path)["sequence"].astype(str).tolist()


def load_latents(path: str | Path = config.LATENTS_NPY) -> torch.Tensor:
    """Load the latent design matrix as a ``[N, LATENT_DIM]`` double tensor in ``[0, 1]``.

    Raises ``ValueError`` if the shape or range is wrong.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"latents file not found: {path}")
    arr = np.load(path)
    latents = torch.as_tensor(arr, dtype=torch.double)
    if latents.ndim != 2 or latents.shape[1] != config.LATENT_DIM:
        raise ValueError(
            f"latents must have shape [N, {config.LATENT_DIM}], got {tuple(latents.shape)}"
        )
    lo, hi = float(latents.min()), float(latents.max())
    if lo < 0.0 or hi > 1.0:
        raise ValueError(f"latents must lie in [0, 1], got range [{lo:.4f}, {hi:.4f}]")
    return latents


def load_initial_ids(path: str | Path = config.INITIAL_IDS_JSON) -> list[int]:
    """Read the fixed initial-design sequence indices from ``initial_indices.json``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"initial indices file not found: {path}")
    payload = json.loads(path.read_text())
    return [int(i) for i in payload["initial_ids"]]


def load_true_objectives(path: str | Path = config.ORACLE_TRUE_NPY) -> torch.Tensor:
    """Load the hidden true-objective table as a ``[N, NUM_OBJECTIVES]`` double tensor.

    Instructor-only -- students reach objectives through the oracle, not this file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"true-objectives file not found: {path}")
    arr = np.load(path)
    objectives = torch.as_tensor(arr, dtype=torch.double)
    if objectives.ndim != 2 or objectives.shape[1] != config.NUM_OBJECTIVES:
        raise ValueError(
            f"true objectives must have shape [N, {config.NUM_OBJECTIVES}], "
            f"got {tuple(objectives.shape)}"
        )
    return objectives
