"""``mobo_lab`` — helper package for the multi-objective Bayesian optimization practicum.

The student-facing notebooks import small, well-documented helpers from here so the
lab can focus on multi-objective decision making rather than BoTorch plumbing. This
top-level module is intentionally tiny: import the submodules you need directly,
e.g. ``from mobo_lab import config, seed``.
"""

__all__ = ["config", "seed"]
