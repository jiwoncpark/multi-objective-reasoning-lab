# Step 14 — Notebook 04: optional extensions

**Status:** DONE (2026-06-28)
**Depends on:** Step 13 (competition module + Notebook 03)
**Unblocks:** nothing (terminal, for faster/advanced groups).

**Result:** `notebooks/04_optional_extensions.{py,ipynb}` — four independent,
heavily-scaffolded sections, each a `competition.run_campaign` vs a baseline with a
comparison table/plot: (1) custom per-round scalarization schedule (via the new
`acquisitions.format_scalarized_name`); (2) explore→exploit vs all-NEHVI; (3)
`nearest` vs `diverse_nearest` on the continuous path; (4) conceptual
information-theoretic discussion + one `uncertainty`-card run. No new core BO code.
Observed contrasts on the real pool: changing weights 0.848 > fixed 0.788 AUC-HV;
diverse_nearest 0.813 > nearest 0.796; uncertainty-first 0.951 AUC-HV (a nice
exploration win). Runs headless in ~50s; `tests/notebooks/test_nb04_smoke.py` green.

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

1. **Custom scalarization schedule (§11.1)** — students supply round-dependent weights and the
   notebook builds the per-round plan from them:
   ```python
   weights_by_round = [[0.5,0.5],[0.8,0.2],[0.2,0.8],[0.5,0.5], ...]
   ```
   Map each to a `scalarized_w1_w2` card (helper formats the name), run via `run_campaign`, compare
   AUC-HV and front coverage to a fixed-weight baseline. Teaching point: matching preferences to
   under-explored regions over time.
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
