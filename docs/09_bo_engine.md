# Step 8 — BO engine: models + acquisitions + optimize

**Status:** DONE (2026-06-28)
**Depends on:** Step 1 (`config`), Step 7 (`pool`)
**Unblocks:** strategies (Step 9), preflight (Step 10), all BO notebooks.

**Result:** three modules verified against **BoTorch 0.18.1**.
`mobo_lab/models.py::fit_surrogate_model` returns a `ModelListGP` of per-objective
`SingleTaskGP`s (Normalize input + Standardize output), optional known-noise via
`train_Yvar`; posterior mean/variance are `[n, NUM_OBJECTIVES]`.
`mobo_lab/acquisitions.py::build_acquisition` is the strategy-card factory for all
seven names — `nehvi` (qLogNEHVI), `parego` (random-weight qLogNParEGO),
`scalarized_{0.5_0.5,0.8_0.2,0.2_0.8}` (fixed-weight qLogNParEGO via
`build_fixed_scalarized_qlognei`), and two `PoolSelector` cards (`random`,
`uncertainty`) that pick pool IDs directly. A seeded `make_sampler` helper feeds
every MC acquisition. `mobo_lab/optimize.py` provides `optimize_continuous`
(taught, `optimize_acqf(sequential=True)`) and `optimize_discrete` (graded,
`optimize_acqf_discrete` over `pool.X` with `X_avoid`); the latter also dispatches
`PoolSelector` cards and returns **exact** IDs (`candidates == pool.X[ids]`) by
identity-projecting the chosen rows. 21 new tests; full mini-loop on the real
2048-row pool (fit → nehvi → discrete → 4 unqueried IDs, plus the continuous +
projection path) runs in ~3s. 141 tests total green.

> **Naming:** the optimizer functions are `optimize_continuous` /
> `optimize_discrete` (docs/09), superseding the outline §12.8 working name
> `optimize_continuous_acquisition`. Outline updated to match.
> **`random`/`uncertainty`:** modelled as `PoolSelector` objects rather than
> BoTorch acquisitions, so `build_acquisition` returns them and `optimize_discrete`
> calls `.select(...)`; `optimize_continuous` rejects them (no continuous form).

## Goal

The heart of the lab: fit a surrogate, build any of the competition acquisition functions by name, and
optimize them into a fixed-size batch via **sequential greedy** — both the continuous path (taught)
and the **discrete pool path** (graded golden path). All GP details stay hidden behind one call
(outline §4.3, §12.6). Signatures below are verified against the installed **BoTorch 0.18.1**.

## Files to create

```text
mobo_lab/models.py
mobo_lab/acquisitions.py
mobo_lab/optimize.py
tests/mobo_lab/test_models.py
tests/mobo_lab/test_acquisitions.py
tests/mobo_lab/test_optimize.py
```

## `mobo_lab/models.py`

```python
def fit_surrogate_model(train_X: Tensor, train_Y: Tensor, train_Yvar: Tensor | None = None):
    """Independent-objective surrogate: a ModelListGP of per-objective SingleTaskGPs."""
```

```python
from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import SumMarginalLogLikelihood

models = []
for k in range(train_Y.shape[-1]):
    yvar = None if train_Yvar is None else train_Yvar[:, k:k+1]
    models.append(SingleTaskGP(train_X, train_Y[:, k:k+1], train_Yvar=yvar,
                               input_transform=Normalize(d=train_X.shape[-1]),
                               outcome_transform=Standardize(m=1)))
model = ModelListGP(*models)
fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))
return model
```

- `ModelListGP` (not a batched single GP) matches the "independent objectives" framing and the
  optional per-objective known-noise path (`train_Yvar`, outline §4.3).
- Wrap `fit_gpytorch_mll` to surface but tolerate `OptimizationWarning` (only 12 initial points).

## `mobo_lab/acquisitions.py`

```python
def build_acquisition(name: str, model, train_X, train_Y, ref_point, sampler, **kwargs):
    """Factory for every strategy card. Returns a ready-to-optimize acquisition (or a
    finite-set selector for non-BoTorch cards like 'random'/'uncertainty')."""
```

Names (outline §12.7) and their construction:

| name | construction (BoTorch 0.18.1) |
|---|---|
| `nehvi` | `qLogNoisyExpectedHypervolumeImprovement(model, ref_point=ref_point, X_baseline=train_X, sampler=sampler, prune_baseline=True)` |
| `parego` | `qLogNParEGO(model, X_baseline=train_X, sampler=sampler, prune_baseline=True)` (random-simplex Chebyshev) |
| `scalarized_0.5_0.5` / `_0.8_0.2` / `_0.2_0.8` | `qLogNParEGO(model, X_baseline=train_X, scalarization_weights=Tensor(w), sampler=sampler, prune_baseline=True)` — fixed Chebyshev weights |
| `random` | not a BoTorch acq; returns a sentinel handled by the optimizer/strategy (samples available pool IDs) |
| `uncertainty` | optional; posterior-stdev-after-scalarization selector over the pool (finite-set), outline §5.4 |

Imports:
```python
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.parego import qLogNParEGO
```

- **Why scalarized = fixed-weight `qLogNParEGO`:** identical, tested code path as ParEGO (Chebyshev
  scalarization wrapped in a `GenericMCObjective` over `qLogNoisyExpectedImprovement`); ParEGO and the
  scalarized cards then differ *only* by random-vs-fixed weights — the cleanest pedagogical contrast,
  and Chebyshev (unlike a linear weighted sum) can still reach concave front regions. (`scalarization_weights`
  must be a `double` tensor; helper `build_fixed_scalarized_qlognei(model, train_X, weights, sampler)`
  wraps this.)
- Parse weights from the name (`scalarized_0.8_0.2` → `[0.8, 0.2]`).

## `mobo_lab/optimize.py`

```python
def optimize_continuous(acq_func, bounds, q=config.BATCH_SIZE,
                        num_restarts=config.NUM_RESTARTS, raw_samples=config.RAW_SAMPLES,
                        sequential=True) -> tuple[Tensor, Tensor]:        # candidates [q,d]
    # botorch.optim.optimize.optimize_acqf(..., options={"batch_limit":5,"maxiter":200}, sequential=True)

def optimize_discrete(acq_func, pool, q=config.BATCH_SIZE,
                      observed_ids=(), pending_ids=()) -> tuple[Tensor, list[int]]:
    # botorch.optim.optimize.optimize_acqf_discrete(acq, q=q, choices=pool.X, unique=True,
    #     X_avoid=pool.X[list(observed_ids)+list(pending_ids)])  -> candidates + their pool ids
```

- `optimize_continuous` is the taught headline (`sequential=True` ⇒ greedy batch fill with internal
  `set_X_pending` conditioning). `optimize_discrete` is the reproducible graded path; its result rows
  are pool rows, so the returned IDs are exact.
- Construct the `SobolQMCNormalSampler` **after** `set_all_seeds` (or pass `seed=config.SEED`) — the
  sampler feeds every acquisition value.

## Tests (small, fast, CPU)

- `test_models.py`: `fit_surrogate_model` on `12×5` X, `12×2` Y returns a `ModelListGP`;
  `model.posterior(test_X).mean.shape == [n, 2]`; accepts `train_Yvar`.
- `test_acquisitions.py`: for each of `nehvi`, `parego`, `scalarized_0.5_0.5`, the acq forward on a
  `[1, q, d]` tensor returns shape `[1]` (the `batch_shape × q × d` convention, outline §8.7);
  `random` returns the sentinel; weight parsing is correct.
- `test_optimize.py`: `optimize_continuous` returns `[q, d]` within `[0,1]`; `optimize_discrete`
  returns `q` distinct IDs, none in `observed_ids`, and the candidate rows equal `pool.X[ids]`.

## Acceptance criteria

- `uv run pytest tests/mobo_lab/test_models.py tests/mobo_lab/test_acquisitions.py
  tests/mobo_lab/test_optimize.py` green on CPU.
- A full mini-loop (fit → build `nehvi` → `optimize_discrete` over the real pool → 4 valid unqueried
  IDs) runs in seconds.
