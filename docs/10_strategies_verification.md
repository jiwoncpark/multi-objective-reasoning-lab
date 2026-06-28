# Step 9 — Strategy cards + verification (skeleton)

**Status:** DONE (2026-06-28)
**Depends on:** Step 8 (engine), Step 7 (pool)
**Unblocks:** preflight (Step 10), Notebook 02 (Step 12), competition (Step 13). Golden constants are
filled in later (Step 11).

**Result:** `mobo_lab/strategies.py` turns a batch plan into a validated batch.
`validate_batch_plan` enforces non-empty, known cards, non-negative int counts,
and sum == `batch_size` with student-friendly messages.
`propose_batch_from_plan` fills the batch card-by-card, extending a `pending` list
so later cards condition on earlier picks; `PoolSelector` cards
(`random`/`uncertainty`) dispatch to `.select`, BoTorch cards to
`optimize_discrete` (default, reproducible) or `optimize_continuous` + projection.
It re-asserts the §10 invariants (size, distinct, none observed) before returning.
`mobo_lab/verification.py::verify_golden_path` compares projected IDs / observed
Y / HVs against placeholder `EXPECTED_*` constants (frozen in Step 11), checks HV
monotonicity, prints the success line, and raises a clear "not frozen yet" error
until the constants are set. 17 new tests; all four real-pool plans (nehvi,
nehvi+parego, dual-scalarized, random+uncertainty) yield 4 distinct unqueried
IDs. 158 tests total green.

> **Signature notes:** added `sampler=None` (auto via `make_sampler`),
> `bounds=None` (auto via `config.latent_bounds()`), and `seed` (for the `random`
> card) as defaulted params so notebooks/tests can omit them; `ref_point` defaults
> to `config.REF_POINT`. The `verify_golden_path` HV-monotonicity guard is only
> reachable when the frozen HV pair itself decreases (exact-match makes it
> otherwise redundant) — tested by pinning a decreasing expected pair.

## Goal

Turn a human-readable **batch plan** (e.g. `{"nehvi": 2, "parego": 2}`) into one validated batch of
`BATCH_SIZE` distinct, unqueried sequence IDs, using sequential greedy per card and conditioning later
picks on earlier pending ones (outline §9, §12.9). Also stand up the `verify_golden_path` checker
whose expected constants are frozen in Step 11.

## Files to create

```text
mobo_lab/strategies.py
mobo_lab/verification.py
tests/mobo_lab/test_strategies.py
tests/mobo_lab/test_verification.py
```

## `mobo_lab/strategies.py`

```python
def validate_batch_plan(batch_plan: dict[str, int], batch_size: int = config.BATCH_SIZE) -> None:
    """Raise a student-friendly ValueError unless sum(values)==batch_size and all names are known."""

def propose_batch_from_plan(batch_plan, model, pool, observed_ids, train_X, train_Y,
                            ref_point, bounds, sampler,
                            batch_size=config.BATCH_SIZE,
                            projection_method=config.PROJECTION_METHOD,
                            optimize="discrete") -> list[int]:
    """Fill the batch one card at a time; return BATCH_SIZE distinct unqueried IDs."""
```

Algorithm (outline §12.9):

1. `validate_batch_plan(batch_plan, batch_size)`.
2. `pending_ids = []`.
3. For each `(name, k)` in the plan:
   - `random` → sample `k` IDs from `pool.available_ids(observed_ids, pending_ids)`.
   - else → `acq = build_acquisition(name, model, train_X, train_Y, ref_point, sampler)`;
     fill `k` slots via sequential greedy:
     - **discrete** (default): `optimize_discrete(acq, pool, q=k, observed_ids, pending_ids)` → IDs.
     - **continuous**: `optimize_continuous(acq, bounds, q=k, sequential=True)` →
       `pool.project_to_unqueried_sequences(cands, observed_ids, pending_ids, projection_method)`.
   - extend `pending_ids` with the new IDs (so subsequent cards condition on them).
4. Assert `len(pending_ids) == batch_size` and all IDs distinct, none in `observed_ids`; return them.

- Known card names come from `acquisitions` + `{"random","uncertainty"}`; surface a clear list in the
  validation error.
- `optimize="discrete"` is the default (reproducible, used by the graded competition); `"continuous"`
  is available for the syntax demos and Notebook 02 exploration.

## `mobo_lab/verification.py`

```python
# Frozen in Step 11 after the golden path is finalized:
EXPECTED_CANDIDATE_IDS: list[int] = []          # filled in Step 11
EXPECTED_NEW_Y: list[list[float]] = []          # filled in Step 11
EXPECTED_HV_BEFORE: float = 0.0                 # filled in Step 11
EXPECTED_HV_AFTER: float = 0.0                  # filled in Step 11

def verify_golden_path(candidate_ids, new_Y, hv_before, hv_after) -> None:
    """Assert IDs == expected; new_Y close (rtol=1e-5, atol=1e-6); HVs close; hv_after >= hv_before.
    On success print: 'Golden-path check passed. You are ready for the strategy-card notebook.'"""
```

Compares projected **IDs / Y / HV** only — never continuous candidates (outline §14 mitigation #5).

## Tests

- `test_strategies.py`:
  - `validate_batch_plan({"nehvi":3}, 4)` raises with a message mentioning the required sum;
    `{"unknown":4}` raises naming valid cards.
  - On a small fitted model + real (or fixture) pool, `propose_batch_from_plan({"nehvi":2,"parego":2})`
    returns **4 distinct** IDs, none observed; `{"random":4}` returns 4 available IDs; a mixed plan with
    pending conditioning never repeats an ID across cards.
  - Both `optimize="discrete"` and `"continuous"` satisfy the invariants.
- `test_verification.py`: `verify_golden_path` passes when given the expected values and raises on a
  mismatched ID / a too-different `new_Y` / `hv_after < hv_before`. (Uses placeholder constants until
  Step 11; test is structured so Step 11 only swaps the numbers.)

## Acceptance criteria

- `uv run pytest tests/mobo_lab/test_strategies.py tests/mobo_lab/test_verification.py` green.
- `propose_batch_from_plan` upholds the §10 anti-confusion invariants (fixed batch size, distinct IDs,
  no re-evaluation of observed sequences) for every default card and mixed plan.
