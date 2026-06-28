# Step 6 — Synthetic oracle + initial design

**Status:** DONE (2026-06-28)
**Depends on:** Step 5 (`vh_latents.npy`)
**Unblocks:** the BO engine (Step 8), strategies/competition, and every notebook that queries objectives.

**Result:** `mobo_lab/oracle.py` + `scripts/build_oracle.py` + `scripts/build_initial_design.py`
materialize `data/oracle_true_objectives.npy` `[2048, 2] ⊂ [0,1]`, `data/oracle_params.json`
(tunable design constants), and `data/initial_indices.json` (12 ids), all byte-identical on
re-run. On the real latents: `spearman(obj1, obj2) = -0.21` (small / mild trade-off), true Pareto
front = **7 points** spanning a max pairwise latent distance of **1.0** (≥2 separated regions). The
initial design is 12 sequences drawn from the **central-or-below** objective band (both objectives
≤ the 60th percentile, `--max-quantile`), then farthest-point-sampled within that band for latent
spread — 0 on the true front; `HV(initial) = 0.409` vs `HV(true front) = 1.025` → **40% of the
achievable max**, i.e. ~60% headroom for the competition. 16 tests green.

> **Preflight note (Step 10):** remaining difficulty knob to revisit at the gate — the front is
> fairly convex (concavity is mild), tunable via `oracle_params.json` without code changes. The
> initial-design headroom is now generous (40% of max); `--max-quantile` can retune it if the
> preflight shows strategies separate too early or too late.
>
> **Update (2026-06-28): GATE PASSED.** The Step 10 preflight passed all §18.4 criteria on the
> current oracle/initial design (see `docs/11`). The earlier "criterion 5 passes narrowly" note is
> **resolved and was not a concavity problem**: deepening the central valley does not widen the
> ParEGO-vs-fixed gap (the angular-spread margin is seed-noise; over-deepening even shrinks the
> front to 5 points). The gate now scores criteria 4 & 5 on **region coverage**, which is the stable
> cross-seed discriminator (ParEGO ≥ fixed in every seed). So this concavity knob is left **as-is**;
> the oracle and the frozen golden constants are unchanged.

## Goal

Define the hidden two-objective **oracle** that maps each library sequence to noisy observations of
two maximization objectives ("binding-like", "stability-like"), plus the fixed **initial design**
(`N_INITIAL` sequence IDs) shared by all teams. The oracle is a **fully synthetic, smooth function of
the 5-D latents**, tuned so the competition is interesting (outline §18.2), with **deterministic,
re-query-stable noise** so the golden path reproduces exactly.

## Files to create

```text
scripts/build_oracle.py             # synthetic objectives over latents -> data/oracle_true_objectives.npy
scripts/build_initial_design.py     # diverse, non-front-saturating starter set -> data/initial_indices.json
mobo_lab/oracle.py                  # AntibodyOracle (evaluate / evaluate_true / deterministic noise)
tests/scripts/test_build_oracle.py
tests/scripts/test_build_initial_design.py
tests/mobo_lab/test_oracle.py
```

## Synthetic objective design (`scripts/build_oracle.py`)

Compute `Y_true = f(X_latent)` where `X_latent` is `data/vh_latents.npy`, with two smooth components
combined to hit the §18.2 difficulty criteria:

```python
# conceptual; tune constants in the script, validate in preflight (Step 10)
g1(x) = <a1, x> + bumps1(x)      # binding-like, driven by one latent direction
g2(x) = <a2, x> + bumps2(x)      # stability-like, driven by a near-orthogonal direction
```

- **Near-independent trade-off:** choose `a1`, `a2` nearly orthogonal so that, evaluated at the `N`
  real latent rows, `spearman(Y_true[:,0], Y_true[:,1]) ≈ 0` (target the ~0.1 "genuine trade-off" of
  the original EDA, now as a *design target*, not a data fact).
- **Multiple Pareto regions:** add 2–3 small Gaussian "bonus bumps" at chosen latent locations so the
  true front has ≥2 separated regions → fixed scalarization can over-focus, exploration can pay off
  (§18.2.3, §18.4.4).
- **Mild front concavity:** a small rotation/curvature term so Chebyshev scalarization (Step 8)
  visibly beats a linear weighted sum — motivates the §11 extension.
- **Normalize** each objective to roughly `[0,1]` over the library so `REF_POINT=[-0.05,-0.05]` sits
  just below the worst real values (outline §16.2.4). Save `Y_true` as `[N, 2]` float64.

Write `data/oracle_true_objectives.npy` (hidden / instructor). Optionally also dump a tunable-params
JSON so the preflight can sweep difficulty without editing code.

## `mobo_lab/oracle.py`

```python
class AntibodyOracle:
    def __init__(self, true_objectives: Tensor, noise_sigma=config.NOISE_SIGMA, seed=config.SEED,
                 allow_true: bool = False): ...
    def evaluate(self, ids: list[int]) -> Tensor:        # [q, 2] noisy observations
    def evaluate_true(self, ids: list[int]) -> Tensor:   # [q, 2]; raises unless allow_true=True

    @classmethod
    def from_files(cls, ...) -> "AntibodyOracle": ...
    @property
    def true_objectives(self) -> Tensor: ...             # instructor-only (allow_true gate)
```

### Deterministic noise (reproducibility-critical)

- At construction, draw **one** fixed noise matrix from a seeded generator:
  `E = rng.standard_normal((N, 2))` with `np.random.default_rng(seed)` (or a seeded
  `torch.Generator`), scale per objective by `noise_sigma`, and set
  `observed = true + E * sigma` once. `evaluate(ids)` is then a **pure lookup** `observed[ids]`.
- Consequence: re-querying the same sequence returns the **same** value (per-sequence noise, outline
  §19.5) — document this to students as a deliberate simplification. HV is therefore reproducible.
- `evaluate_true` / `true_objectives` are gated behind `allow_true` so students cannot peek during the
  campaign (outline §10 coding tasks 1–2); the competition notebook constructs the oracle with
  `allow_true=False`, the instructor reveal with `allow_true=True`.

## `scripts/build_initial_design.py`

- Choose `N_INITIAL=12` sequence IDs that are **diverse in latent space** (e.g. farthest-point /
  k-means medoids over `vh_latents.npy`) and **do not over-seed the Pareto front** (avoid picking many
  already-non-dominated points, outline §18.3.4) — check against `Y_true`'s front and resample if too
  many initial points are non-dominated.
- Write `data/initial_indices.json`: `{"seed": SEED, "initial_ids": [...]}`.

## Tests

- `test_oracle.py`: `evaluate(ids)` deterministic across calls and re-query-stable; `evaluate != true`
  (noise present) but within a few sigma; objectives broadly in `[0,1]`; `evaluate_true` raises when
  `allow_true=False` and returns the stored array when `True`; shapes `[q,2]`.
- `test_build_oracle.py`: synthetic `Y_true` shape `[N,2]`; `|spearman(obj1,obj2)|` below a small
  threshold (near-independent); ≥2 non-dominated points lie in separated latent regions.
- `test_build_initial_design.py`: 12 unique ids; not all on the front; deterministic given seed.

## Acceptance criteria

- `uv run python scripts/build_oracle.py && uv run python scripts/build_initial_design.py` produce the
  two data files reproducibly.
- `uv run pytest tests/scripts/test_build_oracle.py tests/scripts/test_build_initial_design.py
  tests/mobo_lab/test_oracle.py` green.
- Difficulty targets (near-independent objectives, ≥2 front regions) hold — finalized by the Step 10
  preflight gate before any golden values are frozen.
