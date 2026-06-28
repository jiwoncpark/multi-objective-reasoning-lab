# Step 14 — Notebook 04: optional extensions

**Status:** DONE (2026-06-28)
**Depends on:** Step 13 (competition module + Notebook 03)
**Unblocks:** nothing (terminal, for faster/advanced groups).

**Result:** `notebooks/04_optional_extensions.{py,ipynb}` — four independent,
heavily-scaffolded sections, each `competition.run_campaign` based. No new core BO
code. Sections:

1. **A fixed weight commits you to a side of the front** — binding (`scalarized_0.8_0.2`),
   stability (`scalarized_0.2_0.8`), and a rotating schedule, scored by **lean** =
   mean(obj1 − obj2) of the selected points, with a 3-panel objective-space scatter.
   This is the *robust* demonstration of the concavity/weight-steering point: the
   weight reliably steers **which side** of the concave front the picks land on
   (binding lean +0.02, stability lean −0.23, rotating −0.04 in between; holds in all
   5 seeds checked). Earlier framings around AUC-HV or region *count* were dropped —
   `qLogNParEGO` is improvement-driven, so weight moves the *direction* (lean), not
   the score or the region count, which are seed-noise.
2. **Explore→exploit vs all-NEHVI** (HV-curve overlay).
3. **`nearest` vs `diverse_nearest`** on the continuous path.
4. Conceptual information-theoretic discussion + one `uncertainty`-card run
   (uncertainty-then-NEHVI ~0.96 AUC-HV, a standout exploration win).

Sections 2–4 are single-seed illustrations (markdown framed open-ended); section 1's
lean effect is the robust one. Runs headless in ~50s; `tests/notebooks/test_nb04_smoke.py` green.

> **Supporting changes (this step):** `acquisitions.format_scalarized_name` (inverse
> of `parse_scalarized_weights`) and `is_known_card`, and `validate_batch_plan` now
> accepts **any** well-formed `scalarized_<w1>_<w2>` card (not just the three named
> ones) so custom weight schedules validate. `competition.run_campaign` gained an
> `optimize="discrete"|"continuous"` keyword so section 3 can exercise projection on
> the continuous path. 187 tests total green.

## Goal

A clearly-optional notebook for groups who finish early. It reuses the existing engine and competition
machinery — **no new core code is required** — to explore strategy design more deeply (outline §11).
Everything is heavily scaffolded; students should not have to debug BO internals.

## File to create

```text
notebooks/04_optional_extensions.ipynb
```

## Sections (each self-contained; do as many as time allows)

1. **Scalarization weight steers which side of the front (§11.1)** — three campaigns from the same
   start: fixed `scalarized_0.8_0.2` (binding), fixed `scalarized_0.2_0.8` (stability), and a rotating
   schedule alternating the two (built via `format_scalarized_name`). Scored by **lean** =
   mean(obj1 − obj2) of the selected points, with a 3-panel objective-space scatter. Teaching point:
   on a concave front the weight reliably commits a fixed campaign to one **side**, while changing the
   weight over rounds visits both. (Built as the robust replacement for the original "changing-vs-fixed,
   compare AUC-HV/coverage" framing: with the improvement-driven `qLogNParEGO`, weight moves the
   *direction* (lean), not AUC-HV or region count — those are seed-noise. See the result note above.)
2. **Explore→exploit schedule (§11.2)** — the staged plan
   `[{"random":2,"parego":2}, {"parego":4}, {"nehvi":2,"parego":2}, {"nehvi":4}, ...]`; compare to
   `all_nehvi`. Teaching point: spending early budget on exploration can pay off later.
3. **Alternative projection rules (§11.3)** — rerun the same continuous-path campaign with
   `projection_method ∈ {"nearest","diverse_nearest"}` (and note cluster-aware / similarity-penalty as
   stretch ideas); compare coverage and #non-dominated. Teaching point: projection shapes which
   designs actually get tested.
4. **Information-theoretic acquisitions (§11.4, conceptual)** — markdown-only discussion (or a single
   heavily-scaffolded `uncertainty` card run); explicitly framed as "beyond the core lab," not for
   debugging.

## Implementation notes

- Each section calls `competition.run_campaign` / `strategies.propose_batch_from_plan` with different
  inputs; the only new helper is a tiny `weights → "scalarized_x_y"` name formatter (can live in
  `strategies` or inline in the notebook).
- Keep sections independent so a group can jump to whichever interests them.
- Mark the whole notebook "optional / advanced" at the top; nothing here is required for the
  competition or the debrief.

## Tests / acceptance

- `tests/notebooks/test_nb04_smoke.py` (optional): headless execution of at least sections 1–3
  completes on CPU without error.
- Acceptance: each section runs end-to-end and produces a comparison (table or plot) against a
  baseline; no section requires editing `mobo_lab/` core modules.
