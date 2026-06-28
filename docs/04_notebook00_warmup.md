# Step 3 — Notebook 00: Pareto & hypervolume warmup

**Status:** DONE (2026-06-27) — `notebooks/00_*.{py,ipynb}` + smoke test; executes clean, full suite 35/35.
**Depends on:** Steps 1–2 (`config`, `metrics`, `plotting`)
**Unblocks:** student intuition for everything after; it is the first student-facing artifact and an
early end-to-end test of the metrics/plotting stack (no oracle/model/pool needed).

> **Notebook convention (established here, used for 00–04):** author each notebook as a **jupytext
> percent `.py`** (the diffable source of truth) paired to a generated `.ipynb` via the header
> `formats: ipynb,py:percent`. Regenerate the `.ipynb` with `uv run jupytext --sync notebooks/NN_*.py`.
> Ship the `.ipynb` **without** executed outputs (students run it). Smoke-test by reading the `.py`
> with `jupytext` and executing via `nbclient` on the `python3` kernel (`tests/notebooks/`,
> `@pytest.mark.slow`). Dev deps `jupytext`, `nbclient` added to `pyproject.toml`.

## Goal

Introduce multi-objective optimization **before** BoTorch acquisition machinery appears: dominance,
non-dominated points, the Pareto front, the reference point, and hypervolume — and *why* HV grows as
the front expands. Pure warmup on a tiny toy dataset; runs in seconds on CPU.

## File to create

```text
notebooks/00_pareto_hypervolume_warmup.ipynb
```

Build/author it as a `.py` percent-script or via `jupytext`/`nbformat` so it is diffable, then export
to `.ipynb`. (Pick one notebook-authoring convention now and reuse it for Notebooks 01–04.)

## Notebook structure (cells)

1. **Narrative markdown** — the wet-lab framing (outline §15): two properties, trade-offs, no single
   best antibody, so we seek the Pareto front. No mention of `docs/` (CLAUDE.md rule).
2. **Imports + seed** — `from mobo_lab import config, seed, metrics, plotting`; `seed.set_all_seeds()`.
3. **Toy data** — the §7 `Y_toy` `[6, 2]` tensor, presented as "six candidate antibodies, two scores."
4. **Task: scatter** — `plotting.plot_objective_space(Y_toy)`; student eyeballs trade-offs.
5. **Task: dominance** — short student exercise to mark which points are dominated; reveal with
   `metrics.compute_pareto_mask(Y_toy)`; overlay via `plot_objective_space(Y_toy, pareto_mask=mask)`.
6. **Task: hypervolume** — set `ref_point = config.REF_POINT` (and a toy `[0,0]`); compute
   `metrics.compute_hypervolume`; shade the dominated region in the plot.
7. **Task: move the reference point** — recompute HV for two ref points; discuss why a more
   pessimistic ref point increases HV (outline §7.4).
8. **Task: add a point** — append one new candidate; show HV increases only if it is non-dominated
   (outline §7.5); reuse `plot_pareto_front` with the new point highlighted.
9. **Wrap-up markdown** — define the five takeaways (dominance, non-dominated set, Pareto front, ref
   point, hypervolume); forward-reference Notebook 01 without naming internal docs.

## Implementation notes

- Everything routes through `mobo_lab.metrics` / `mobo_lab.plotting`; the notebook contains **no**
  hand-rolled dominance/HV math (students learn the concepts, the library owns the implementation).
- Keep student "TODO" cells minimal and immediately followed by a reveal cell so groups can't get
  stuck (this is a guided warmup, not an assessment).
- Re-state `REF_POINT`/seed as visible literals next to the `config` import (pedagogy, outline §8.3).

## Tests / acceptance

- Smoke-execute headless on CPU:
  `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/00_pareto_hypervolume_warmup.ipynb`
  completes with no errors.
- The Pareto mask cell prints the four non-dominated toy points; the "add a point" cell shows
  `hv_after >= hv_before`.
- (Optional) a tiny `tests/notebooks/test_nb00_smoke.py` that runs the nbconvert execution and asserts
  exit status 0, so notebook rot is caught by CI alongside module tests.
