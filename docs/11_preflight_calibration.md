# Step 10 — Instructor preflight: difficulty calibration (GATE)

**Status:** DONE — GATE PASSED (2026-06-28)
**Depends on:** Step 9 (strategies), Step 8 (engine), Steps 4–6 (data assets)
**Unblocks:** the golden-path freeze (Step 11) and the competition (Step 13).
**This is a hard gate:** golden values are not frozen and the lab is not taught until preflight passes.
A failure loops back to Steps 5–6 (retune embedding/oracle) or Step 4 (re-curate the library).

**Result:** `scripts/preflight_sweep.py` sweeps 7 strategies × 3 seeds × 6 rounds on
the real 2048-sequence pool and **all §18.4 criteria PASS** (run 2026-06-28):

| metric (mean) | AUC-HV | finalHV | angSpr | cover | nondom |
|---|---|---|---|---|---|
| all_nehvi | 0.8503 | 1.0147 | 0.248 | 4.7 | 7.0 |
| all_scalarized_0.8_0.2 | 0.8474 | 0.9761 | 0.112 | 2.0 | 4.7 |
| all_parego | 0.8358 | 0.9853 | 0.119 | 2.7 | 4.0 |
| explore_then_exploit | 0.8292 | 1.0038 | 0.221 | 4.7 | 6.0 |
| scalarization_sweep | 0.8288 | 0.9971 | 0.173 | 3.0 | 5.0 |
| mixed | 0.8280 | 1.0057 | 0.210 | 4.0 | 6.0 |
| random_baseline | 0.6840 | 0.8071 | 0.268 | 5.0 | 2.3 |

Highlights: nehvi clearly beats random (0.850 vs 0.684); fixed scalarization
concentrates (angSpr 0.112 vs nehvi 0.248); a mixed strategy beats all_nehvi in
some seeds (mixed 2/3, parego 2/3, explore 1/3); leaderboard not predetermined
(2 distinct per-seed winners, top-two mean gap 0.0029); full campaign in ~5.6s;
discrete acq margin q-vs-(q+1) abs 0.037 (rel 9.3e-4); achieved-vs-true at 99% of
max HV. Plots in `outputs/preflight/{hv_curves,selection_coverage}.png`. Shrunk
test (`tests/scripts/test_preflight_sweep.py`) green; 165 tests total.

> **Watch-item for Step 11 / Step 6:** criterion 5 (ParEGO explores more than
> fixed scalarization) passes **narrowly** — angSpr 0.119 vs 0.112. The front is
> only mildly concave, so random ParEGO weights don't spread selections much more
> than a fixed weight. It is safe to freeze, but if a wider ParEGO signal is
> wanted, increase the oracle front concavity / bump separation in Step 6 and
> re-run the gate. The discrete-path reproducibility (which Notebook 01 relies on)
> is unaffected.

## Goal

Before freezing anything, verify offline that the curated library + latents + synthetic oracle +
initial design produce a **competition worth running** (outline §18): strategies must diverge, model-
guided must beat random, scalarization must concentrate, and runtime must fit the schedule.

## File to create

```text
scripts/preflight_sweep.py
tests/scripts/test_preflight_sweep.py     # runs a shrunk sweep on a fixture to prove it executes
```

## What it does

Run the §18.1 candidate strategies under the **same** initial design, budget, and batch size, using
the real engine (`run_campaign`-style loop, or `competition.run_campaign` once Step 13 exists — the
preflight may import it or carry a local loop to stay ahead of Step 13):

```python
strategies_to_test = {
  "all_nehvi":          [{"nehvi": 4}] * N_ROUNDS,
  "all_parego":         [{"parego": 4}] * N_ROUNDS,
  "scalarization_sweep":[{"scalarized_0.8_0.2":2,"scalarized_0.2_0.8":2}, ..., {"nehvi":4}],
  "explore_then_exploit":[{"random":2,"parego":2}, {"parego":4}, ...],
  "mixed":              [{"nehvi":2,"parego":2}, {"nehvi":3,"random":1}, ...],
  "random_baseline":    [{"random": 4}] * N_ROUNDS,
}
```

For each strategy record (§18.1): AUC-HV, final HV, per-round HV gain, #non-dominated selected,
selected-embedding diversity, and which objective-space regions were covered. Use a fixed seed for the
oracle noise; optionally average over a few campaign seeds for the strategy-internal randomness
(ParEGO weights, random card) and report mean±sd.

## §18.4 acceptance criteria (must all pass)

1. Golden-path inputs are deterministic (re-run identical).
2. `nehvi` beats `random_baseline` in AUC-HV (on average).
3. At least one **mixed** strategy sometimes beats `all_nehvi`.
4. Fixed scalarization visibly **concentrates** in one objective-space region (report a coverage/
   spread metric for `scalarized_0.8_0.2` vs `nehvi`).
5. ParEGO explores different trade-offs across rounds (spread of selected points).
6. The leaderboard is not predetermined by one obvious strategy (no single strategy dominates all).
7. A full 6-round campaign runs comfortably within the practicum schedule (time it; target well under
   a minute per campaign on CPU).
8. Plots are interpretable.
9. (Checked at the reveal step) achieved-vs-true front contrast is meaningful.

Also assert the **top-q discrete acquisition margin**: in the golden-path round, the 4th-best pool
point's acq value is clearly separated from the 5th — so the discrete argmax is robust to float-level
differences across machines (protects Notebook 01 reproducibility; relates to the qLogNEHVI fused-vs-
pure-Python kernel caveat).

## Outputs

- A console summary table + a few PNGs under `outputs/preflight/` (HV curves per strategy, coverage
  scatter). These are instructor artifacts, not student-facing.
- A clear PASS/FAIL banner per §18.4 criterion. On FAIL, print which knob to turn (oracle bumps/
  rotation in Step 6, embedding spread in Step 5, library coverage in Step 4, or budget in `config`).

## Implementation notes

- Keep the sweep parameterized so difficulty knobs (oracle params JSON from Step 6) can be adjusted and
  re-swept without code edits.
- This script may be slow-ish (many campaigns); the **test** runs a shrunk version (2 strategies, 2
  rounds, tiny pool fixture) just to prove it executes and the metrics compute.

## Acceptance criteria

- `uv run python scripts/preflight_sweep.py` prints all §18.4 criteria as PASS, with the discrete-margin
  check passing, before Step 11 proceeds.
- `uv run pytest tests/scripts/test_preflight_sweep.py` green (shrunk run).
