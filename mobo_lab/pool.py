"""The finite candidate pool the campaign optimizes over.

``VHSequencePool`` bundles the three views of the curated library that the BO loop
needs and keeps them aligned by a single integer index:

* ``X``         -- the ``[N, LATENT_DIM]`` latent design matrix in ``[0, 1]``,
* ``sequences`` -- the ``N`` amino-acid strings, and
* ``ids``       -- ``0 .. N-1``, the row index shared by both of the above.

Because an ID *is* a row index, ``pool.X[ids]`` and ``[pool.sequences[i] for i in
ids]`` line up with no extra bookkeeping, and the oracle (which is indexed the same
way) can be queried with the very IDs projection returns (outline §8.3).

The pool exposes the two operations the loop relies on:

* :meth:`available_ids` -- the IDs still in play (not observed, not pending), used
  by the ``"random"`` strategy and to build ``optimize_acqf_discrete``'s
  ``X_avoid``; and
* :meth:`project_to_unqueried_sequences` -- turn a batch of continuous acquisition
  proposals into distinct, unqueried IDs (delegates to :mod:`mobo_lab.projection`).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import torch
from torch import Tensor

from . import config, data, projection


class VHSequencePool:
    """A finite pool of VH sequences with aligned latents and integer IDs.

    Parameters
    ----------
    X:
        ``[N, LATENT_DIM]`` latent design matrix; coerced to ``double``.
    sequences:
        The ``N`` amino-acid strings, in the same row order as ``X``.

    Raises
    ------
    ValueError
        If ``X`` is not ``[N, LATENT_DIM]``, its rows do not match ``sequences``,
        or its values fall outside ``[0, 1]``.
    """

    def __init__(self, X: Tensor, sequences: list[str]) -> None:
        X = torch.as_tensor(X, dtype=torch.double)
        sequences = list(sequences)
        if X.ndim != 2 or X.shape[1] != config.LATENT_DIM:
            raise ValueError(
                f"pool X must have shape [N, {config.LATENT_DIM}], got {tuple(X.shape)}"
            )
        if X.shape[0] != len(sequences):
            raise ValueError(
                f"pool has {X.shape[0]} latent rows but {len(sequences)} sequences"
            )
        lo, hi = float(X.min()), float(X.max())
        if lo < 0.0 or hi > 1.0:
            raise ValueError(f"pool X must lie in [0, 1], got range [{lo:.4f}, {hi:.4f}]")

        self.X = X
        self.sequences = sequences
        self.ids: list[int] = list(range(X.shape[0]))

    # -- construction ------------------------------------------------------- #
    @classmethod
    def from_files(
        cls,
        library_csv: str | Path = config.LIBRARY_CSV,
        latents_npy: str | Path = config.LATENTS_NPY,
    ) -> "VHSequencePool":
        """Build the pool from the curated library CSV and the latents ``.npy``."""
        sequences = data.load_sequences(library_csv)
        X = data.load_latents(latents_npy)
        return cls(X, sequences)

    def __len__(self) -> int:
        return len(self.ids)

    # -- queries ------------------------------------------------------------ #
    def available_ids(
        self,
        observed_ids: Iterable[int],
        pending_ids: Iterable[int] | None = None,
    ) -> list[int]:
        """Return the IDs not yet observed or pending, in ascending order."""
        forbidden = self._forbidden(observed_ids, pending_ids)
        return [i for i in self.ids if i not in forbidden]

    def project_to_unqueried_sequences(
        self,
        candidates: Tensor,
        observed_ids: Iterable[int],
        pending_ids: Iterable[int] | None = None,
        method: str = config.PROJECTION_METHOD,
    ) -> list[int]:
        """Map ``[q, d]`` continuous proposals to ``q`` distinct, unqueried IDs.

        ``forbidden = observed_ids | pending_ids`` is built and handed to the
        chosen projection strategy (``"nearest"`` or ``"diverse_nearest"``), which
        guarantees the returned IDs are distinct and none of them observed or
        pending. See :mod:`mobo_lab.projection` for the strategies.

        Raises
        ------
        KeyError
            If ``method`` is not a known projection strategy.
        ValueError
            If the pool has too few available rows to fill the batch.
        """
        if method not in projection.METHODS:
            raise KeyError(
                f"unknown projection method {method!r}; "
                f"choices: {sorted(projection.METHODS)}"
            )
        forbidden = self._forbidden(observed_ids, pending_ids)
        return projection.METHODS[method](candidates, self.X, forbidden)

    # -- internals ---------------------------------------------------------- #
    @staticmethod
    def _forbidden(
        observed_ids: Iterable[int], pending_ids: Iterable[int] | None
    ) -> set[int]:
        forbidden = {int(i) for i in observed_ids}
        if pending_ids is not None:
            forbidden |= {int(i) for i in pending_ids}
        return forbidden
