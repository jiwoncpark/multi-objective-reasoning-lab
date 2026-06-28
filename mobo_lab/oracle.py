"""The hidden two-objective oracle the wet-lab "experiments" are queried through.

``AntibodyOracle`` wraps the instructor-only true-objective table
(``data/oracle_true_objectives.npy``, built by ``scripts/build_oracle.py`` as a
smooth synthetic function of the latents) and exposes the **noisy** observations
students actually see.

Reproducibility is the whole point of this class. The observation noise is drawn
**once** at construction from a seeded generator and baked into a fixed
``observed = true + noise`` table, so:

* ``evaluate(ids)`` is a pure lookup -- re-querying the same sequence returns the
  *same* value (deterministic per-sequence noise, a deliberate teaching
  simplification), which keeps hypervolume curves reproducible; and
* the golden-path notebook reproduces exactly across student machines.

The true objectives are gated behind ``allow_true`` so students cannot peek during
the campaign: the competition notebook builds the oracle with ``allow_true=False``;
the instructor reveal builds it with ``allow_true=True``.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch

from . import config, data


class AntibodyOracle:
    """Noisy two-objective oracle over the curated VH library.

    Parameters
    ----------
    true_objectives:
        ``[N, NUM_OBJECTIVES]`` hidden true objective values (maximization).
    noise_sigma:
        Per-objective observation-noise standard deviation.
    seed:
        Seeds the one-shot noise draw, so the observed table is reproducible.
    allow_true:
        If ``False`` (default), :meth:`evaluate_true` and :attr:`true_objectives`
        raise -- the student-facing safety gate.
    """

    def __init__(
        self,
        true_objectives: torch.Tensor | np.ndarray,
        noise_sigma: tuple[float, ...] = config.NOISE_SIGMA,
        seed: int = config.SEED,
        allow_true: bool = False,
    ) -> None:
        true = torch.as_tensor(true_objectives, dtype=torch.double)
        if true.ndim != 2 or true.shape[1] != config.NUM_OBJECTIVES:
            raise ValueError(
                f"true_objectives must have shape [N, {config.NUM_OBJECTIVES}], "
                f"got {tuple(true.shape)}"
            )
        if len(noise_sigma) != config.NUM_OBJECTIVES:
            raise ValueError(
                f"noise_sigma must have {config.NUM_OBJECTIVES} entries, got {len(noise_sigma)}"
            )

        self._true = true
        self._allow_true = bool(allow_true)
        self._noise_sigma = tuple(float(s) for s in noise_sigma)

        # Draw the noise ONCE, deterministically, and bake the observed table.
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal(size=tuple(true.shape)) * np.asarray(self._noise_sigma)
        self._observed = true + torch.as_tensor(noise, dtype=torch.double)

    # -- construction ------------------------------------------------------- #
    @classmethod
    def from_files(
        cls,
        true_path: str | Path = config.ORACLE_TRUE_NPY,
        noise_sigma: tuple[float, ...] = config.NOISE_SIGMA,
        seed: int = config.SEED,
        allow_true: bool = False,
    ) -> "AntibodyOracle":
        """Build the oracle from ``data/oracle_true_objectives.npy``."""
        true = data.load_true_objectives(true_path)
        return cls(true, noise_sigma=noise_sigma, seed=seed, allow_true=allow_true)

    # -- queries ------------------------------------------------------------ #
    def evaluate(self, ids: Iterable[int]) -> torch.Tensor:
        """Return noisy observations ``[q, NUM_OBJECTIVES]`` for ``ids`` (pure lookup)."""
        idx = self._as_index(ids)
        return self._observed[idx].clone()

    def evaluate_true(self, ids: Iterable[int]) -> torch.Tensor:
        """Return noiseless true objectives ``[q, NUM_OBJECTIVES]``; instructor-only."""
        self._require_true("evaluate_true")
        idx = self._as_index(ids)
        return self._true[idx].clone()

    @property
    def true_objectives(self) -> torch.Tensor:
        """The full hidden true-objective table ``[N, NUM_OBJECTIVES]``; instructor-only."""
        self._require_true("true_objectives")
        return self._true.clone()

    def __len__(self) -> int:
        return self._true.shape[0]

    # -- internals ---------------------------------------------------------- #
    def _require_true(self, what: str) -> None:
        if not self._allow_true:
            raise PermissionError(
                f"{what} is instructor-only; construct AntibodyOracle(..., allow_true=True) "
                "to access the hidden true objectives (students must not peek during the campaign)."
            )

    def _as_index(self, ids: Iterable[int]) -> torch.Tensor:
        idx = torch.as_tensor(list(ids), dtype=torch.long)
        if idx.ndim != 1:
            raise ValueError(f"ids must be a 1-D sequence of indices, got shape {tuple(idx.shape)}")
        return idx
