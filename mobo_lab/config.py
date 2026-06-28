"""Shared configuration for the multi-objective Bayesian optimization lab.

This module is the single importable source of truth for the constants used by the
``mobo_lab`` package, the instructor scripts, and the notebooks. The same numbers
also appear as visible literals inside the notebooks (for teaching); keep the two
in sync.

It is deliberately import-light -- only ``pathlib`` and ``torch`` -- so importing it
is cheap and free of the heavier Bayesian-optimization machinery (which, for
BoTorch's fused acquisition kernels, can be slow to import).
"""

from __future__ import annotations

from pathlib import Path

import torch

# --- Reproducibility ---------------------------------------------------------
SEED = 123

# --- Campaign geometry -------------------------------------------------------
BATCH_SIZE = 4          # antibodies tested per wet-lab round (fixed throughput)
N_INITIAL = 12          # sequences evaluated before the campaign begins
N_ROUNDS = 6            # number of design rounds in the competition
TOTAL_NEW_EVALUATIONS = BATCH_SIZE * N_ROUNDS   # 24 new evaluations total

# --- Design and objective spaces --------------------------------------------
LATENT_DIM = 5          # dimensionality of the [0, 1]^d latent design space
NUM_OBJECTIVES = 2      # "binding-like" and "stability-like", both maximized
LIBRARY_SIZE = 2048     # 2^11 curated candidate pool (NN library + ground-truth front)

# --- Acquisition / optimizer -------------------------------------------------
NUM_RESTARTS = 10
RAW_SAMPLES = 128
MC_SAMPLES = 64
REF_POINT = [-0.05, -0.05]   # length == NUM_OBJECTIVES; below any real objective value

# --- Oracle observation noise (per-objective standard deviation) -------------
NOISE_SIGMA = (0.05, 0.05)

# --- Defaults ----------------------------------------------------------------
PROJECTION_METHOD = "nearest"
PRIMARY_SCORE = "auc_hv"
TIE_BREAKER = "final_hv"

# --- Filesystem (resolved relative to the repo root, never the CWD) ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUTS_DIR = REPO_ROOT / "outputs"
LIBRARY_CSV = DATA_DIR / "vh_library.csv"
LATENTS_NPY = DATA_DIR / "vh_latents.npy"
INITIAL_IDS_JSON = DATA_DIR / "initial_indices.json"
ORACLE_TRUE_NPY = DATA_DIR / "oracle_true_objectives.npy"

# Invariant: the reference point has one entry per objective.
assert len(REF_POINT) == NUM_OBJECTIVES, "REF_POINT must have NUM_OBJECTIVES entries"
assert len(NOISE_SIGMA) == NUM_OBJECTIVES, "NOISE_SIGMA must have NUM_OBJECTIVES entries"


def ref_point_tensor(dtype: torch.dtype = torch.double) -> torch.Tensor:
    """Return ``REF_POINT`` as a tensor of shape ``[NUM_OBJECTIVES]``."""
    return torch.tensor(REF_POINT, dtype=dtype)


def latent_bounds(dtype: torch.dtype = torch.double) -> torch.Tensor:
    """Return the ``[0, 1]^LATENT_DIM`` box as a ``[2, LATENT_DIM]`` tensor.

    Row 0 holds the per-dimension lower bounds (all zeros) and row 1 the upper
    bounds (all ones) -- the layout BoTorch's ``optimize_acqf`` expects for
    ``bounds``.
    """
    return torch.stack(
        [
            torch.zeros(LATENT_DIM, dtype=dtype),
            torch.ones(LATENT_DIM, dtype=dtype),
        ]
    )
