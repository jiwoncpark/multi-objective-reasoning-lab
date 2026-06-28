"""One-call reproducibility helper for the lab.

Seeds Python's ``random``, NumPy, and PyTorch (CPU) from a single function so every
notebook starts from the same state, and switches PyTorch to double precision -- the
golden-path notebook relies on deterministic CPU behavior.

We intentionally do *not* enable ``torch.use_deterministic_algorithms(True)`` by
default: it raises on some ops and is unnecessary here, because cross-machine
reproducibility is achieved through the discrete-pool acquisition path rather than
bit-exact continuous optimization.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from . import config


def set_all_seeds(seed: int = config.SEED) -> None:
    """Seed ``random``, ``numpy``, and ``torch`` (CPU) and default to float64.

    Parameters
    ----------
    seed:
        The integer seed to apply to all three random number generators.
        Defaults to :data:`mobo_lab.config.SEED`.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # The lab runs in double precision on CPU for deterministic behavior.
    torch.set_default_dtype(torch.double)
