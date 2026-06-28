# Step 2 — Metrics + plotting

**Status:** DONE (2026-06-27) — `mobo_lab/{metrics,plotting}.py` + tests; 13/13 green, full suite 34/34.
**Depends on:** Step 1 (`config`)
**Unblocks:** Notebook 00 (Step 3), strategies/competition (Steps 9, 13), and the golden-path checks.

> **Correction (as built):** for the `Y_toy` set, **only `(0.3, 0.3)` is dominated** — `(0.5, 0.5)`
> is non-dominated because no single point beats it on *both* objectives (each anti-diagonal point has
> one coordinate below 0.5). The test asserts mask `[T,T,T,T,T,F]`. `_pareto_staircase` was
> re-implemented inside `plotting.py` (routing dominance through `metrics.compute_pareto_mask`) rather
> than importing from `scripts/`, since `scripts/` is not an importable package.

## Goal

Provide the multi-objective bookkeeping (Pareto mask, hypervolume, AUC-HV, diversity) and the core
plots, both thin wrappers over BoTorch utilities so students trust them as a black box. These are
pure leaves (only depend on BoTorch + matplotlib), so they can be built and tested with no data
assets.

## Files to create

```text
mobo_lab/metrics.py
mobo_lab/plotting.py
tests/mobo_lab/test_metrics.py
```

(No plotting test beyond a smoke "returns an Axes / writes a file"; the math lives in `metrics`.)

## Public API

### `metrics.py`

```python
def compute_pareto_mask(Y: Tensor) -> Tensor:          # [n, m] -> bool [n]; maximization
def compute_hypervolume(Y: Tensor, ref_point) -> float
def compute_auc_hv(hv_history: Sequence[float]) -> float
def compute_embedding_diversity(X_selected: Tensor) -> float   # mean pairwise L2 (or min)
def compute_true_pareto_front(Y_true_all: Tensor) -> Tensor    # returns mask or the front points
```

### `plotting.py`

```python
def plot_objective_space(Y, pareto_mask=None, ref_point=None, ax=None, title=None): ...
def plot_pareto_front(Y, selected_mask=None, ref_point=None, title=None, ax=None): ...
def plot_hv_curve(hv_history, title=None, ax=None): ...
```

(`plot_true_front_with_team_overlays(...)` is specified later with the competition module, Step 13,
since it needs team-run structures.)

## Implementation notes (verified BoTorch 0.18.1)

```python
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning

def compute_pareto_mask(Y):
    return is_non_dominated(Y)                 # maximize=True, deduplicate=True by default

def compute_hypervolume(Y, ref_point):
    ref = torch.as_tensor(ref_point, dtype=Y.dtype)     # MUST be a Tensor here
    return DominatedPartitioning(ref_point=ref, Y=Y).compute_hypervolume().item()
```

- `compute_auc_hv`: trapezoidal integral of `hv_history` over round index, normalized by number of
  intervals (so it's comparable across runs with equal `N_ROUNDS`). Document the convention (outline
  §10 "AUC-HV over rounds").
- `compute_embedding_diversity`: mean pairwise Euclidean distance of `X_selected` (tie-breaker §10.3);
  define for `n < 2` to return `0.0`.
- Reuse the already-tested staircase idea from `scripts/visualize_data.py` (`_pareto_staircase`) for
  drawing the front line in `plot_pareto_front`; import or re-implement the tiny helper — do not
  duplicate the dominance math (use `compute_pareto_mask`).
- All functions accept/return `torch.double` tensors; convert ref points via `torch.as_tensor`.
- Plotting functions accept an optional `ax`, return the `ax`, and never call `plt.show()` (so they
  compose in notebooks and tests). Mark maximization direction on axis labels.

## Tests (`tests/mobo_lab/test_metrics.py`) — intuitive numerical cases

Use the outline §7 toy set:

```python
Y_toy = torch.tensor([[0.2,0.8],[0.4,0.6],[0.6,0.4],[0.8,0.2],[0.5,0.5],[0.3,0.3]], dtype=torch.double)
```

- **Pareto mask:** only `[0.3,0.3]` is dominated; the other five points — the four "anti-diagonal"
  points **and** `[0.5,0.5]` — are non-dominated. Assert mask equals `[T,T,T,T,T,F]`.
- **Hypervolume hand-check:** with `ref_point=[0,0]`, compute HV of a 2–3 point set by hand (sum of
  axis-aligned rectangle areas) and assert `compute_hypervolume` matches to `1e-9`.
- **Monotonicity:** adding a non-dominated point strictly increases HV; adding a dominated point
  leaves it unchanged.
- **Ref-point sensitivity:** a more pessimistic `ref_point` yields larger HV (sanity, outline §7.4).
- **AUC-HV:** for `hv_history=[0,1,2,3]` the normalized trapezoid equals the hand value.
- **Diversity:** two identical rows → `0.0`; an equilateral-ish triple → expected mean distance.

## Acceptance criteria

- `uv run pytest tests/mobo_lab/test_metrics.py` green.
- `compute_pareto_mask` agrees with `is_non_dominated` on random `[n,2]` tensors (property check).
- `plot_pareto_front(Y_toy, ref_point=[0,0])` returns an `Axes` without error.
