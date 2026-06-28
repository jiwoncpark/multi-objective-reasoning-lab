"""Tests for ``mobo_lab/pool.py`` (the finite candidate pool + projection wiring)."""

from __future__ import annotations

import pytest
import torch

from mobo_lab import config
from mobo_lab.pool import VHSequencePool

# A small pool whose rows march along the first latent axis at 0.0, 0.2, ..., 1.0,
# so "nearest" is just "closest first coordinate" -- easy to reason about. The
# remaining four latent dimensions are zero. LATENT_DIM rows are required because
# the pool constructor enforces the [N, LATENT_DIM] contract.
_FIRST_COORDS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def _pool() -> VHSequencePool:
    n = len(_FIRST_COORDS)
    X = torch.zeros(n, config.LATENT_DIM, dtype=torch.double)
    X[:, 0] = torch.tensor(_FIRST_COORDS, dtype=torch.double)
    sequences = [f"SEQ{i}" for i in range(n)]
    return VHSequencePool(X, sequences)


def _along_axis(value: float) -> list[float]:
    """A candidate point at ``value`` on axis 0, zero elsewhere."""
    return [value] + [0.0] * (config.LATENT_DIM - 1)


def test_ids_are_row_indices():
    pool = _pool()
    assert pool.ids == list(range(len(_FIRST_COORDS)))
    assert len(pool) == len(_FIRST_COORDS)


def test_available_ids_excludes_observed_and_pending():
    pool = _pool()
    avail = pool.available_ids(observed_ids=[0, 1], pending_ids=[5])
    assert avail == [2, 3, 4]  # complement, ascending


def test_available_ids_without_pending():
    pool = _pool()
    assert pool.available_ids(observed_ids=[2]) == [0, 1, 3, 4, 5]


def test_project_returns_distinct_unqueried_ids():
    pool = _pool()
    observed, pending = [0, 1], [5]
    # Proposals sit on top of row 0 and row 5, both of which are off-limits, so
    # projection must fall back to the nearest *available* rows.
    candidates = torch.tensor(
        [_along_axis(0.05), _along_axis(0.95)], dtype=torch.double
    )
    ids = pool.project_to_unqueried_sequences(candidates, observed, pending_ids=pending)
    assert ids == [2, 4]
    assert len(set(ids)) == len(ids)
    assert not (set(ids) & (set(observed) | set(pending)))


def test_project_default_method_is_identity_for_exact_rows():
    pool = _pool()
    ids = pool.project_to_unqueried_sequences(pool.X[[1, 3]], observed_ids=[])
    assert ids == [1, 3]


def test_project_diverse_method_dispatches():
    pool = _pool()
    ids = pool.project_to_unqueried_sequences(
        pool.X[[0, 1]], observed_ids=[], method="diverse_nearest"
    )
    assert len(ids) == 2 and len(set(ids)) == 2


def test_project_unknown_method_raises():
    pool = _pool()
    with pytest.raises(KeyError, match="unknown projection method"):
        pool.project_to_unqueried_sequences(
            pool.X[[0]], observed_ids=[], method="teleport"
        )


def test_constructor_rejects_wrong_latent_dim():
    with pytest.raises(ValueError, match="shape"):
        VHSequencePool(torch.zeros(3, config.LATENT_DIM + 1), ["A", "B", "C"])


def test_constructor_rejects_length_mismatch():
    with pytest.raises(ValueError, match="sequences"):
        VHSequencePool(torch.zeros(3, config.LATENT_DIM), ["A", "B"])


def test_constructor_rejects_out_of_range_values():
    X = torch.zeros(2, config.LATENT_DIM, dtype=torch.double)
    X[0, 0] = 1.5
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        VHSequencePool(X, ["A", "B"])


def test_from_files_loads_real_pool():
    # Integration check against the curated assets (Step 4 library + Step 5 latents).
    pool = VHSequencePool.from_files()
    n = len(pool)
    assert n == config.LIBRARY_SIZE
    assert pool.X.shape == (n, config.LATENT_DIM)
    assert pool.X.dtype == torch.double
    assert float(pool.X.min()) >= 0.0 and float(pool.X.max()) <= 1.0
    assert pool.ids == list(range(n))
    assert len(pool.sequences) == n
