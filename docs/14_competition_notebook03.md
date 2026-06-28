# Step 13 — Competition module + Notebook 03 (+ true-front reveal)

**Status:** TODO
**Depends on:** Step 9 (`strategies`), Step 11 (locked assets), Steps 2–8.
**Unblocks:** the live competition and the final debrief; Notebook 04 builds on it.

## Goal

Groups compete to maximize hypervolume under a fixed budget by editing only a per-round strategy plan.
Provide the campaign runner, leaderboard I/O, and the instructor **true-Pareto-front reveal** that
turns the ranking into a scientific discussion (outline §10).

## Files to create

```text
mobo_lab/competition.py
notebooks/03_competition.ipynb
tests/mobo_lab/test_competition.py
```

(`plotting.plot_true_front_with_team_overlays` is added here, since it needs team-run structures.)

## `mobo_lab/competition.py`

```python
def run_campaign(team_strategy: list[dict], team_name: str, seed=config.SEED,
                 projection_method=config.PROJECTION_METHOD) -> dict:
    """N_ROUNDS rounds; each round: fit -> propose_batch_from_plan -> oracle.evaluate -> update ->
    record HV. Returns a history dict (per-round ids, Y, hv, plus auc_hv/final_hv)."""

def save_run_outputs(history: dict, output_dir=config.OUTPUTS_DIR) -> None:   # json + csv + pngs
def load_team_runs(output_dir=config.OUTPUTS_DIR) -> list[dict]
def update_leaderboard(output_dir=config.OUTPUTS_DIR) -> pd.DataFrame         # ranked by AUC-HV
def build_final_debrief_report(output_dir, oracle, initial_ids, ref_point) -> Path
```

- Fixed (not student-editable): `BATCH_SIZE`, `N_ROUNDS`, `N_INITIAL`, the oracle, the initial design,
  the budget (outline §10).
- Student-editable: `TEAM_STRATEGY` (list of per-round plans) and optional `PROJECTION_METHOD`.
- **Anti-confusion guards** enforced inside `run_campaign` (outline §10): fixed batch size every round;
  no duplicate selected IDs; never re-evaluate observed IDs; no access to true objectives during the
  run (oracle built with `allow_true=False`); oracle/initial design/budget immutable.
- Scoring: primary `auc_hv` (`metrics.compute_auc_hv` over the per-round observed HV); tie-breakers —
  final HV, #non-dominated selected, embedding diversity, earliest round to a target HV (outline §10).
- `save_run_outputs` writes `outputs/{team}_run.json` (schema in outline §10), `{team}_history.csv`,
  `{team}_pareto_plot.png`, `{team}_hv_curve.png`.

## Notebook 03 structure

1. **Briefing markdown** — the §10 competition framing (same start, `BATCH_SIZE`/round, `N_ROUNDS`
   rounds, maximize HV; you may change strategy each round).
2. **Locked setup cell** — loads pool/oracle/initial design; constants shown read-only.
3. **The one editable cell** — `TEAM_NAME = "..."`, `TEAM_STRATEGY = [{...}, ...]` (len `N_ROUNDS`),
   optional `PROJECTION_METHOD`. Validate each plan up front.
4. **Run** — `history = run_campaign(TEAM_STRATEGY, TEAM_NAME)`; `save_run_outputs(history)`; show the
   HV curve and the team's achieved Pareto front.
5. **Leaderboard** — `update_leaderboard()` renders the ranked table from all saved runs.
6. **Instructor reveal section** — built/run by the instructor:
   - reconstruct the oracle with `allow_true=True`;
   - `compute_true_pareto_front(oracle.true_objectives)`;
   - `plot_true_front_with_team_overlays(Y_true_all, initial_ids, load_team_runs(), ref_point,
     output_path="outputs/final_true_pareto_overlay.png")` with the §10 visual layers (gray all
     candidates, black true front, outlined initial design, colored per-team achieved fronts, faint
     selected points, marked ref point);
   - markdown discussion prompts (§10): which regions were found/missed, where scalarization over-
     focused, where NEHVI/ParEGO/exploration helped, what projection changed.

## Implementation notes

- `run_campaign` reuses `strategies.propose_batch_from_plan`, `models.fit_surrogate_model`,
  `metrics.*`, `oracle.evaluate` — no new BO logic; it is the multi-round driver around Step 9.
- Keep the true-objective access path strictly behind `allow_true` so a student running cells can't
  reach it; the reveal is a separate, clearly instructor-marked section (outline §10 coding tasks).
- Leaderboard is robust to multiple runs of the same team (latest wins or suffix run ids).

## Tests (`tests/mobo_lab/test_competition.py`)

- `run_campaign` on a tiny fixture (small pool, 2 rounds): returns `N_ROUNDS` rounds; HV history
  non-decreasing; `auc_hv`/`final_hv` computed; **never** selects an observed ID; batch size fixed.
- Anti-confusion: attempting a plan whose round sum ≠ `BATCH_SIZE` raises; oracle with
  `allow_true=False` blocks `evaluate_true`.
- I/O round-trip: `save_run_outputs` then `load_team_runs` recovers the run; `update_leaderboard`
  ranks by AUC-HV; reveal helper writes `final_true_pareto_overlay.png`.

## Acceptance criteria

- `uv run pytest tests/mobo_lab/test_competition.py` green.
- Notebook 03 runs a full 6-round campaign on CPU within the practicum schedule, writes the four team
  artifacts, renders the leaderboard, and (instructor section) produces
  `outputs/final_true_pareto_overlay.png`.
