# Step 11 — Notebook 01: seeded golden-path MOBO iteration

**Status:** TODO
**Depends on:** Step 10 preflight **passing** (assets locked), plus Steps 1–9.
**Unblocks:** Notebooks 02–03; this is the keystone reproducibility milestone.

## Goal

The single most important notebook: one fully seeded, single-iteration "golden path" that every
student reproduces **exactly** (same candidate IDs, observed Y, HV before/after) on CPU, while also
exposing the real BoTorch syntax for the competition acquisitions. Build it only after the preflight
gate (Step 10) locks the oracle/embedding/initial-design, because those determine the frozen values.

## File to create / modify

```text
notebooks/01_seeded_noisy_sequential_greedy_mobo_iteration.ipynb
mobo_lab/verification.py            # fill in the frozen EXPECTED_* constants
```

(Filename reconciled to the §6 outline name; README reference updated to match.)

## Notebook structure (mirrors outline §8, with the graded path made discrete)

- **01.1 Imports + seeding** — visible literals for `SEED/BATCH_SIZE/LATENT_DIM/NUM_RESTARTS/
  RAW_SAMPLES/MC_SAMPLES/REF_POINT`; `seed.set_all_seeds(SEED)`; `device=cpu`, `dtype=double`.
- **01.2 Load pool** — `pool = VHSequencePool.from_files()`; assert `X.ndim==2`,
  `X.shape[1]==LATENT_DIM`, `0 ≤ X ≤ 1`.
- **01.3 Initial design** — `initial_ids = data.load_initial_ids()`; `train_X = pool.X[initial_ids]`;
  `oracle = AntibodyOracle.from_files(allow_true=False)`; `train_Y = oracle.evaluate(initial_ids)`;
  assert shapes `[N_INITIAL, LATENT_DIM]`, `[N_INITIAL, 2]`.
- **01.4 Fit model** — `model = fit_surrogate_model(train_X, train_Y)` (black-box framing).
- **01.5 Sampler** — `SobolQMCNormalSampler(torch.Size([MC_SAMPLES]), seed=SEED)` (constructed right
  after seeding; note why).
- **01.6 Build qLogNEHVI** — the real constructor with shape comments
  (`train_X:[n,d]`, `train_Y:[n,2]`, candidate `X:[batch,q,d]`).
- **01.7 Evaluate acq on a known test batch** — `test_X = pool.X[[0,1,2,3]].unsqueeze(0)` →
  `acq(test_X).shape == [1]` (makes `batch×q×d` concrete).
- **01.8 (GRADED) Optimize over the discrete pool** —
  `candidate_ids = optimize_discrete(acq_nehvi, pool, q=BATCH_SIZE, observed_ids=initial_ids)`'s IDs.
  Markdown explains sequential greedy: slot 1, then slot 2 conditioned on slot 1, … This is the path
  the golden checks grade, because discrete argmax is reproducible across CPUs.
- **01.8b (NON-GRADED) Continuous syntax demo** — show `optimize_continuous(acq_nehvi, bounds,
  q=BATCH_SIZE, sequential=True)` returning `[q, d]`, then
  `pool.project_to_unqueried_sequences(...)`. Clearly labeled: "for syntax; exact IDs may vary by
  machine — the graded result is the discrete cell above." (Outline §14 mitigation #6.)
- **01.9 Projection sanity** — assert `len(candidate_ids)==BATCH_SIZE`, all distinct, none in
  `initial_ids`.
- **01.10 Query oracle** — `new_Y = oracle.evaluate(candidate_ids)`; assert `[BATCH_SIZE, 2]`.
- **01.11 Update + hypervolume** — `updated_Y = cat([train_Y, new_Y])`;
  `hv_before/after = compute_hypervolume(·, REF_POINT)`; assert `hv_after >= hv_before`.
- **01.12 Before/after plots** — `plot_pareto_front` with new candidates highlighted; print HVs.
- **01.13 Acquisition syntax clinic** — construct `nehvi`, `parego`, and a fixed `scalarized_0.5_0.5`
  and show all three flow through the **same** optimizer interface (the §8.13 unification).
- **01.14 Shape-error table** — the §8.14 debugging table as markdown.
- **01.15 Verification cell** — `verify_golden_path(candidate_ids, new_Y, hv_before, hv_after)` →
  prints the success message.

## Freezing the golden constants (the milestone)

1. Run the notebook end-to-end on CPU after Step 10 locks assets.
2. Copy the realized `candidate_ids`, `new_Y`, `hv_before`, `hv_after` into
   `mobo_lab/verification.py` `EXPECTED_*`.
3. Re-run twice on this machine → identical. Then **re-verify on a second clean environment** (fresh
   `uv sync`) before teaching (outline §18.4.1).
4. Residual determinism guards: sampler seeded after `set_all_seeds`; if the preflight discrete-margin
   was tight, optionally force qLogNEHVI's pure-Python path in this notebook for cross-machine safety
   (document if used).

## Tests / acceptance

- `tests/mobo_lab/test_verification.py` updated to assert against the real frozen constants.
- `tests/notebooks/test_nb01_golden.py`: execute the notebook headless
  (`jupyter nbconvert --execute`) and assert it completes and the verification cell's expected values
  match `verification.EXPECTED_*` (guards against drift).
- Acceptance: two consecutive headless runs produce identical IDs/Y/HV; `verify_golden_path` passes;
  confirmed on a second machine.
