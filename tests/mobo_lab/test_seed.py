"""Tests for ``mobo_lab/seed.py``.

Re-seeding with the same value must reproduce identical draws from all three RNGs,
and ``set_all_seeds`` must leave PyTorch in double precision.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from mobo_lab import seed


def test_torch_draws_reproducible():
    seed.set_all_seeds(0)
    a = torch.rand(3)
    seed.set_all_seeds(0)
    b = torch.rand(3)
    assert torch.equal(a, b)


def test_numpy_and_random_draws_reproducible():
    seed.set_all_seeds(0)
    numpy_first = np.random.rand(3)
    random_first = random.random()

    seed.set_all_seeds(0)
    numpy_second = np.random.rand(3)
    random_second = random.random()

    assert np.array_equal(numpy_first, numpy_second)
    assert random_first == random_second


def test_default_dtype_is_double():
    seed.set_all_seeds(0)
    assert torch.get_default_dtype() == torch.float64


def test_default_seed_runs():
    # Calling with no argument uses config.SEED and must not raise.
    seed.set_all_seeds()
