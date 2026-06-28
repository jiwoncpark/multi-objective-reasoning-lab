# Step 12 — Notebook 02: strategy-card practice

**Status:** DONE (2026-06-28)
**Depends on:** Step 9 (`strategies`), Step 11 (Notebook 01 patterns + locked assets)
**Unblocks:** the competition (students need fluency with batch plans first).

**Result:** `notebooks/02_strategy_cards_practice.{py,ipynb}` reuses the Notebook 01
setup, introduces the batch-plan dict + `validate_batch_plan`, runs one round from a
plan, then compares six plans (`nehvi`, `parego`, `scalarized_0.8_0.2`,
`scalarized_0.2_0.8`, `nehvi+parego`, `nehvi+random`) **all from the same observed
state** — tabulating selected IDs, HV gain, and an objective-space spread
(`metrics.compute_embedding_diversity` on the selected objectives), with a 2×3
selection-scatter grid, an HV-improvement bar chart, and discussion prompts. Adds
no new BO logic — pure orchestration over `propose_batch_from_plan` with
`optimize="discrete"`. Each plan's invariants (BATCH_SIZE distinct unqueried IDs,
HV non-decreasing) are asserted in-notebook. Runs headless in ~6s;
`tests/notebooks/test_nb02_smoke.py` green; 169 tests total.

## Goal

Show that the same closed-loop machinery is driven by different acquisition choices via a simple
**batch plan**. Still guided (not yet a competition): students run a round or two with different plans
and compare the resulting batches, objective scatter, and HV improvement (outline §9).

## File to create

```text
notebooks/02_strategy_cards_practice.ipynb
```

## Notebook structure

1. **Recap markdown** — strategy cards as "campaign strategies" (outline §15): NEHVI expands the
   front, ParEGO samples random preferences, fixed scalarization commits to a preference, random/
   uncertainty spend budget on exploration.
2. **Setup** — `seed.set_all_seeds()`; load pool + oracle (`allow_true=False`) + initial design; fit
   model once (reuse the Notebook 01 setup verbatim so students see continuity).
3. **The batch-plan object** — show the dict form and the sum rule:
   ```python
   batch_plan = {"nehvi": 4}                 # or {"nehvi":2,"parego":2}
   strategies.validate_batch_plan(batch_plan)   # friendly error if sum != BATCH_SIZE
   ```
4. **Run one round from a plan** —
   `ids = strategies.propose_batch_from_plan(batch_plan, model, pool, observed_ids, train_X, train_Y,
   ref_point, bounds, sampler)`; `new_Y = oracle.evaluate(ids)`; update; recompute HV.
5. **Compare cards** — loop over a small set of plans (`{"nehvi":4}`, `{"parego":4}`,
   `{"scalarized_0.8_0.2":4}`, `{"scalarized_0.2_0.8":4}`, `{"nehvi":2,"parego":2}`,
   `{"nehvi":3,"random":1}`), each starting from the **same** observed set, and tabulate: selected IDs,
   HV improvement, whether selected points cluster in one region of objective space (outline §9 tasks).
6. **Plots** — overlay each plan's selected points on the objective scatter; one HV-improvement bar
   chart across plans. Highlight that `scalarized_0.8_0.2` concentrates while `nehvi`/`parego` spread.
7. **Discussion markdown** — short prompts: which plan grew HV most this round? which concentrated?
   why might that change over multiple rounds? (Sets up the competition's multi-round trade-offs.)

## Implementation notes

- Default `optimize="discrete"` for reproducible, fast comparisons; mention `"continuous"` as the
  alternative shown in Notebook 01.
- All cards reuse `strategies.propose_batch_from_plan` — the notebook adds **no** new BO logic, only
  orchestration and plotting, keeping student focus on *strategy*, not implementation.
- Keep each comparison a **single round from a common state** (not a full campaign) so it's fast and
  the effect of the card choice is isolated.

## Tests / acceptance

- `tests/notebooks/test_nb02_smoke.py`: headless execution completes; each plan yields `BATCH_SIZE`
  distinct unqueried IDs; HV is non-decreasing after each plan's update.
- Acceptance: notebook runs top-to-bottom on CPU in well under a minute; the comparison table shows
  visibly different selections across cards (sanity that the cards aren't collapsing to the same set).
