# Multi-Objective Bayesian Optimization Practicum: Curriculum + Coding-Agent Outline

## 1. Core teaching premise

Students are acting as a wet-lab design team.

They have a library of antibody VH sequences. Each sequence can be experimentally evaluated for two noisy objectives. In each round, the wet lab can evaluate a fixed batch of antibodies. Students must choose which antibodies to evaluate in order to improve the Pareto front and maximize hypervolume.

The lab emphasizes:

1. Multi-objective trade-offs.
2. Pareto optimality and hypervolume.
3. Noisy acquisition functions used to construct fixed-size batches via sequential greedy optimization.
4. The BoTorch closed-loop pattern:
   - fit/update surrogate,
   - construct acquisition function,
   - construct each fixed-size batch using sequential greedy acquisition optimization,
   - map continuous candidates to valid sequences,
   - evaluate oracle,
   - update Pareto front and hypervolume.
5. Strategy design under a fixed wet-lab batch size, where each batch is filled one candidate at a time using sequential greedy optimization.

The lab intentionally de-emphasizes:

1. Teaching Gaussian process theory.
2. Asking students to train sequence models.
3. Asking students to design custom acquisition functions from scratch.
4. Open-ended latent-space decoding of arbitrary generated sequences.

---

## 2. High-level structure of the 3-hour practicum

Recommended schedule:

| Time | Segment | Main output |
|---:|---|---|
| 0:00–0:15 | Motivation and setup | Students understand the antibody design campaign |
| 0:15–0:35 | Pareto and hypervolume warmup | Students compute/visualize Pareto front and HV |
| 0:35–1:20 | Notebook 01: one fully seeded noisy sequential-greedy batch BO iteration | Students reproduce expected candidate IDs and HV |
| 1:20–1:35 | BoTorch syntax clinic | Students inspect acquisition constructors and tensor shapes |
| 1:35–1:45 | Break | — |
| 1:45–2:10 | Strategy-card practice | Students compare prebuilt strategies |
| 2:10–2:45 | Mini-competition | Groups run fixed-budget campaign |
| 2:45–3:00 | Final reveal and debrief | Reveal the true Pareto front, overlay group results, and discuss where different acquisition strategies helped |

The schedule can be compressed by merging the syntax clinic into Notebook 01.

---

## 3. Conceptual setup

### 3.1 Design objects

The biological design objects are antibody VH sequences.

Each sequence has:

- a unique sequence ID,
- an amino-acid sequence string,
- a precomputed latent embedding `x in [0, 1]^d`,
- hidden objective values available only through the oracle.

The main lab should treat the VH sequence library as a finite candidate pool.

### 3.2 Latent representation

The coding agent should prepare a latent design matrix:

```python
X_pool: torch.Tensor  # shape [num_sequences, latent_dim]
```

Recommended dimensions:

```python
latent_dim = 8
```

The latent vectors should be scaled to `[0, 1]^d`, so that BoTorch can use standard box constraints:

```python
bounds = torch.stack([
    torch.zeros(latent_dim, dtype=torch.double),
    torch.ones(latent_dim, dtype=torch.double),
])
```

### 3.3 Continuous proposal + valid-sequence projection

BoTorch will optimize acquisition functions in continuous latent space. However, the actual evaluated objects are valid VH sequences from the library.

Therefore, every acquisition proposal should follow:

```text
continuous latent candidate z*
    -> nearest valid unevaluated sequence in library
    -> sequence ID
    -> oracle evaluation
```

The projection step should be implemented in helper code and should handle:

1. Already observed sequence IDs.
2. Duplicate continuous candidates that map to the same sequence.
3. Diversity-aware nearest-neighbor projection, if enabled.
4. Fallback behavior when a nearest sequence has already been selected in the current batch.

Recommended API:

```python
candidate_ids = pool.project_to_unqueried_sequences(
    candidates=candidates,          # shape [q, d]
    observed_ids=observed_ids,
    pending_ids=pending_ids,
    method="diverse_nearest",
)
```

For the first version, `method="nearest"` is sufficient. The competition can optionally expose `method="diverse_nearest"` as a strategy knob.

---

## 4. Objectives and oracle

### 4.1 Objective convention

Use two objectives and make both maximization objectives.

Example framing:

```text
Objective 1: binding-like score, higher is better
Objective 2: developability-like score, higher is better
```

The exact biological names can be changed later.

#### Note (data-driven, 2026-06-27): which properties actually trade off

The raw library `data/vh_data.xlsx` (113 VH sequences) carries four *simulated*
biophysical properties. We do not use them directly, but the EDA in
`scripts/visualize_data.py` tells us how to orient the synthetic oracle:

- Re-expressed as "higher is better" (`pKd = -log10(Kd[M])`, `Tm`, `Yield`,
  `-BV`), the Spearman correlations show **binding (pKd) is essentially
  uncorrelated with stability/yield** (rho ~ 0.14 / 0.04), while **Tm and Yield
  are strongly redundant** (rho ~ 0.81).
- Therefore the natural two-objective pair is **binding vs stability** (a genuine
  near-independent trade-off), with yield/BV available as flavor or as a third
  objective in extensions. Building the oracle around two redundant properties
  would collapse the Pareto front to a near-diagonal and make the competition
  trivial (cf. §18.2 difficulty criteria).

### 4.2 Noisy observations

The practicum should use noisy observations, because the acquisition functions will be the noisy variants.

The oracle should expose:

```python
Y_obs = oracle.evaluate(candidate_ids)
```

where:

```python
Y_obs.shape == [q, 2]
```

Optionally, the oracle can also have a hidden true objective:

```python
Y_true = oracle.evaluate_true(candidate_ids)
```

But students should not call this during the campaign. The instructor/leaderboard may use the true objectives for final scoring if desired.

### 4.3 Observation noise handling

The starter model can assume known observation noise, inferred noise, or a small fixed noise variance. Since the lab is not about GP modeling, the coding agent should hide most model details behind:

```python
model = fit_surrogate_model(train_X, train_Y, train_Yvar=None)
```

If known noise is used:

```python
train_Yvar = torch.full_like(train_Y, noise_variance)
```

or per-objective:

```python
noise_variances = torch.tensor([sigma1**2, sigma2**2], dtype=torch.double)
train_Yvar = noise_variances.expand_as(train_Y)
```

---

## 5. Recommended acquisition functions

All required acquisition functions should be noisy and batch-capable, but the notebooks should construct batches using sequential greedy optimization rather than full joint batch optimization.

Sequential greedy means that a batch of size `q = BATCH_SIZE` is filled one slot at a time. At each greedy step, the acquisition function is optimized for the next candidate while conditioning on the candidates already chosen for the current pending batch. This matches the lecture plan and is easier to explain than full joint optimization over all `q` candidates at once.

### 5.1 Primary method: qLogNEHVI

Use qLogNoisyExpectedHypervolumeImprovement as the main method.

Pedagogical description:

```text
qLogNEHVI scores a whole batch by the expected improvement in hypervolume,
while accounting for uncertainty/noise in the currently observed Pareto front.
```

Implementation concept:

```python
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)

acq_func = qLogNoisyExpectedHypervolumeImprovement(
    model=model,
    ref_point=ref_point,
    X_baseline=train_X,
    sampler=sampler,
    prune_baseline=True,
)
```

Then optimize using sequential greedy batch construction:

```python
candidates, acq_value = optimize_acqf(
    acq_function=acq_func,
    bounds=bounds,
    q=BATCH_SIZE,
    num_restarts=NUM_RESTARTS,
    raw_samples=RAW_SAMPLES,
    options={"batch_limit": 5, "maxiter": 200},
    sequential=True,
)
```

### 5.2 Random scalarization method: qLogNParEGO

Use qLogNParEGO as the randomized scalarization method.

Pedagogical description:

```text
qLogNParEGO turns the multi-objective problem into a sequence of scalarized
noisy improvement problems, using random trade-off weights.
```

Implementation concept:

```python
from botorch.acquisition.multi_objective.parego import qLogNParEGO

acq_func = qLogNParEGO(
    model=model,
    X_baseline=train_X,
    sampler=sampler,
    prune_baseline=True,
)
```

Alternatively, if more explicit control over scalarization weights is desired, expose a helper that samples or sets scalarization weights.

### 5.3 Fixed scalarization + noisy batch improvement

Use a fixed-weight scalarization as the simplest preference-based comparator.

Pedagogical description:

```text
Fixed scalarization encodes a specific preference over objectives before
optimization. It is easy to understand but may cover only one part of the
Pareto front.
```

Possible implementation:

```python
weights = torch.tensor([0.5, 0.5], dtype=torch.double)

# Helper should build a scalarized noisy improvement acquisition.
acq_func = build_fixed_scalarized_qlognei(
    model=model,
    train_X=train_X,
    weights=weights,
    sampler=sampler,
)
```

The coding agent should choose the most stable current BoTorch implementation, likely using a scalarized objective or posterior transform with a noisy log expected improvement acquisition.

Expose weights through simple strategy names:

```python
"scalarized_0.5_0.5"
"scalarized_0.8_0.2"
"scalarized_0.2_0.8"
```

### 5.4 Optional exploration strategy

A pure information-theoretic acquisition function is probably too much for the core 3-hour lab. Instead, expose an exploration slot as a simple strategy card.

Possible options:

1. Posterior standard deviation after scalarization.
2. Thompson sampling over a finite candidate set.
3. Max posterior uncertainty among projected candidate pool points.
4. Random unevaluated sequence baseline.

This should be a strategy-card option, not a core concept.

---

## 6. Notebook layout

The final deliverable should include the following notebooks.

```text
notebooks/
  00_pareto_hypervolume_warmup.ipynb
  01_seeded_noisy_sequential_greedy_mobo_iteration.ipynb
  02_strategy_cards_practice.ipynb
  03_competition.ipynb
  04_optional_extensions.ipynb
```

---

## 7. Notebook 00: Pareto and hypervolume warmup

### Purpose

Introduce multi-objective optimization before BoTorch appears.

Students should understand:

1. Dominance.
2. Non-dominated points.
3. Pareto front.
4. Reference point.
5. Hypervolume.
6. Why hypervolume increases when the front expands.

### Inputs

Use a tiny toy dataset:

```python
Y_toy = torch.tensor([
    [0.2, 0.8],
    [0.4, 0.6],
    [0.6, 0.4],
    [0.8, 0.2],
    [0.5, 0.5],
    [0.3, 0.3],
], dtype=torch.double)
```

### Tasks

Students should:

1. Plot objective 1 vs objective 2.
2. Identify non-dominated points.
3. Compute hypervolume.
4. Change the reference point and observe the effect.
5. Add one new point and see whether HV changes.

### Coding-agent tasks

Implement:

```python
plot_objective_space(Y, pareto_mask=None, ref_point=None)
compute_pareto_mask(Y)
compute_hypervolume(Y, ref_point)
```

Expected BoTorch utilities to use:

```python
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
```

---

## 8. Notebook 01: Seeded noisy sequential-greedy batch MOBO iteration

### Purpose

This is the most important notebook. It should be a fully seeded, single-iteration "golden path" that students can verify.

Students should run one complete noisy batch MOBO step, constructed using sequential greedy optimization, and get identical outputs.

This notebook should also explicitly expose students to BoTorch syntax for:

1. qLogNEHVI.
2. qLogNParEGO.
3. Fixed scalarization + noisy batch improvement.
4. `optimize_acqf`.
5. Sequential greedy optimization via `sequential=True`.
6. Batch size `q`.
7. MC sampler construction.
8. `X_baseline`.
9. Reference point.
10. Tensor shapes.

### Global constants

```python
SEED = 123
BATCH_SIZE = 4
LATENT_DIM = 5
NUM_RESTARTS = 10
RAW_SAMPLES = 128
MC_SAMPLES = 64
REF_POINT = torch.tensor([-0.05, -0.05], dtype=torch.double)
```

Use CPU for deterministic behavior in the golden-path notebook:

```python
device = torch.device("cpu")
torch.set_default_dtype(torch.double)
```

### Section 01.1: Imports and seeding

Students should see:

```python
import random
import numpy as np
import torch

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
```

If needed:

```python
torch.use_deterministic_algorithms(False)
```

The coding agent should test the exact environment and document whether strict determinism is feasible.

### Section 01.2: Load sequence pool

Expected objects:

```python
pool = VHSequencePool.from_files(...)
oracle = AntibodyOracle.from_files(...)

X_pool = pool.X
sequences = pool.sequences
```

Assertions:

```python
assert X_pool.ndim == 2
assert X_pool.shape[1] == LATENT_DIM
assert X_pool.min() >= 0.0
assert X_pool.max() <= 1.0
```

### Section 01.3: Load initial design

Use fixed initial sequence IDs.

```python
initial_ids = load_initial_ids("data/initial_indices.json")
train_X = pool.X[initial_ids]
train_Y = oracle.evaluate(initial_ids)
```

Assertions:

```python
assert train_X.shape == (N_INITIAL, LATENT_DIM)
assert train_Y.shape == (N_INITIAL, 2)
```

### Section 01.4: Fit model

Expose the call but hide GP details:

```python
model = fit_surrogate_model(train_X, train_Y)
```

For students:

```text
We will treat the model as a black-box uncertainty engine. The practicum focuses
on the acquisition function and the closed-loop optimization logic.
```

For coding agent:

- Implement the model in a stable way.
- Prefer a simple independent-objective model.
- Keep input normalization and outcome standardization inside the helper.
- Return a fitted BoTorch model compatible with all acquisition functions.

### Section 01.5: Construct MC sampler

Students should see:

```python
from botorch.sampling.normal import SobolQMCNormalSampler

sampler = SobolQMCNormalSampler(
    sample_shape=torch.Size([MC_SAMPLES])
)
```

Explain:

```text
The acquisition function is approximated with Monte Carlo samples from the
model's posterior.
```

### Section 01.6: Build qLogNEHVI explicitly

Students should see the real BoTorch constructor:

```python
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)

acq_func = qLogNoisyExpectedHypervolumeImprovement(
    model=model,
    ref_point=REF_POINT.tolist(),
    X_baseline=train_X,
    sampler=sampler,
    prune_baseline=True,
)
```

Required shape comments:

```python
# train_X: [n_observed, d]
# train_Y: [n_observed, 2]
# candidate batch X passed to acq_func: [batch_shape, q, d] or [q, d]
```

### Section 01.7: Evaluate acquisition on a test batch

Before optimizing, students should evaluate the acquisition function on a known test batch.

```python
test_X = pool.X[[0, 1, 2, 3]].unsqueeze(0)  # shape [1, q, d]
value = acq_func(test_X)

assert value.shape == torch.Size([1])
```

This makes the `batch_shape x q x d` convention concrete.

### Section 01.8: Optimize acquisition with fixed q using sequential greedy

Students should see:

```python
from botorch.optim.optimize import optimize_acqf

bounds = torch.stack([
    torch.zeros(LATENT_DIM, dtype=torch.double),
    torch.ones(LATENT_DIM, dtype=torch.double),
])

candidates, acq_value = optimize_acqf(
    acq_function=acq_func,
    bounds=bounds,
    q=BATCH_SIZE,
    num_restarts=NUM_RESTARTS,
    raw_samples=RAW_SAMPLES,
    options={"batch_limit": 5, "maxiter": 200},
    sequential=True,
)
```

Assertions:

```python
assert candidates.shape == (BATCH_SIZE, LATENT_DIM)
```

Pedagogical note:

```text
Here q equals the number of antibodies selected in the wet-lab batch.
With sequential=True, BoTorch fills the batch greedily: candidate 1, then
candidate 2 conditional on candidate 1, and so on until the batch is full.
This is different from full joint batch optimization over all q candidates at once.
```

### Section 01.9: Project continuous candidates to valid VH sequences

```python
candidate_ids = pool.project_to_unqueried_sequences(
    candidates=candidates,
    observed_ids=initial_ids,
    method="nearest",
)
```

Assertions:

```python
assert len(candidate_ids) == BATCH_SIZE
assert len(set(candidate_ids)) == BATCH_SIZE
assert not any(idx in initial_ids for idx in candidate_ids)
```

Golden-path check:

```python
expected_candidate_ids = [...]
assert candidate_ids == expected_candidate_ids
```

The coding agent must fill in the expected IDs after the dataset/oracle is fixed.

### Section 01.10: Query noisy oracle

```python
new_Y = oracle.evaluate(candidate_ids)
```

Assertions:

```python
assert new_Y.shape == (BATCH_SIZE, 2)
```

Golden-path check:

```python
expected_new_Y = torch.tensor([...], dtype=torch.double)
torch.testing.assert_close(new_Y, expected_new_Y, rtol=1e-5, atol=1e-6)
```

The coding agent must fill in the expected values after the oracle is fixed.

### Section 01.11: Update data and hypervolume

```python
updated_X = torch.cat([train_X, pool.X[candidate_ids]], dim=0)
updated_Y = torch.cat([train_Y, new_Y], dim=0)

hv_before = compute_hypervolume(train_Y, REF_POINT)
hv_after = compute_hypervolume(updated_Y, REF_POINT)
```

Golden-path check:

```python
assert abs(hv_before - EXPECTED_HV_BEFORE) < 1e-6
assert abs(hv_after - EXPECTED_HV_AFTER) < 1e-6
assert hv_after >= hv_before
```

### Section 01.12: Plot before/after Pareto front

Produce:

1. Objective scatter before.
2. Objective scatter after.
3. Pareto front highlighted.
4. Newly selected candidates highlighted.
5. Hypervolume before/after printed.

### Section 01.13: Compare acquisition syntax

This section should expose the syntax for all competition-relevant acquisitions without requiring students to run a full campaign yet.

#### qLogNEHVI

```python
acq_nehvi = qLogNoisyExpectedHypervolumeImprovement(
    model=model,
    ref_point=REF_POINT.tolist(),
    X_baseline=train_X,
    sampler=sampler,
    prune_baseline=True,
)
```

#### qLogNParEGO

```python
from botorch.acquisition.multi_objective.parego import qLogNParEGO

acq_parego = qLogNParEGO(
    model=model,
    X_baseline=train_X,
    sampler=sampler,
    prune_baseline=True,
)
```

#### Fixed scalarization

Use helper:

```python
acq_scalarized = build_fixed_scalarized_qlognei(
    model=model,
    train_X=train_X,
    weights=torch.tensor([0.5, 0.5], dtype=torch.double),
    sampler=sampler,
)
```

Then show that all acquisition functions can be sent through a common sequential-greedy optimizer interface:

```python
candidates, acq_value = optimize_acqf(
    acq_function=acq_func,
    bounds=bounds,
    q=BATCH_SIZE,
    num_restarts=NUM_RESTARTS,
    raw_samples=RAW_SAMPLES,
    options={"batch_limit": 5, "maxiter": 200},
    sequential=True,
)
```

### Section 01.14: Common shape errors

Add a small debugging table:

| Object | Intended shape | Common error |
|---|---:|---|
| `train_X` | `[n, d]` | accidentally `[n, 1, d]` |
| `train_Y` | `[n, m]` | using `[m, n]` |
| `test_X` for acq eval | `[batch_shape, q, d]` | missing q dimension |
| `candidates` from `optimize_acqf` | `[q, d]` | treating q as MC samples |
| `sequential=True` | fills `[q, d]` greedily | confusing greedy batch construction with full joint q-dimensional optimization |
| `Y_new` | `[q, m]` | forgetting objective dimension |
| `X_baseline` | `[n, d]` | passing all pool candidates instead of observed candidates |

### Section 01.15: End-of-notebook verification cell

The notebook should end with a single summary check:

```python
verify_golden_path(
    candidate_ids=candidate_ids,
    new_Y=new_Y,
    hv_before=hv_before,
    hv_after=hv_after,
)
```

Expected output:

```text
Golden-path check passed.
You are ready for the strategy-card notebook.
```

---

## 9. Notebook 02: Strategy-card practice

### Purpose

Students now see that the same closed-loop machinery can be driven by different acquisition choices and projection rules. All non-random acquisition strategies should fill their allocated batch slots using sequential greedy optimization.

This notebook should still be guided. It should not be a full competition yet.

### Strategy-card abstraction

Students should manipulate a simple batch-plan object:

```python
batch_plan = {
    "nehvi": 4,
}
```

or:

```python
batch_plan = {
    "nehvi": 2,
    "parego": 2,
}
```

or:

```python
batch_plan = {
    "scalarized_0.8_0.2": 2,
    "scalarized_0.2_0.8": 2,
}
```

### Required strategy cards

Implement these:

1. `"nehvi"`: qLogNEHVI.
2. `"parego"`: qLogNParEGO.
3. `"scalarized_0.5_0.5"`: fixed balanced scalarization.
4. `"scalarized_0.8_0.2"`: objective-1-favoring scalarization.
5. `"scalarized_0.2_0.8"`: objective-2-favoring scalarization.
6. `"random"`: random unevaluated sequence baseline.
7. Optional: `"uncertainty"`: exploration-heavy candidate.

### Fixed batch size

All plans must satisfy:

```python
sum(batch_plan.values()) == BATCH_SIZE
```

If not, raise a student-friendly error.

### Practice tasks

Students should run one or two rounds with different batch plans and compare:

1. Candidate IDs.
2. Objective scatter.
3. HV improvement.
4. Whether the selected candidates cluster in one part of objective space.

---

## 10. Notebook 03: Competition

### Purpose

Groups compete to maximize hypervolume under a fixed evaluation budget. The batch size is fixed, but acquisition-based batches are constructed using sequential greedy optimization rather than full joint batch optimization.

### Competition framing

```text
You all start with the same initial evaluated antibodies.
Each round, your wet lab can test exactly BATCH_SIZE new antibodies.
Within each round, acquisition-based strategies fill the batch greedily, one slot at a time.
You have N_ROUNDS rounds.
You may choose a different strategy plan each round.
Your goal is to maximize hypervolume as quickly as possible.
```

### Fixed values

Recommended defaults:

```python
BATCH_SIZE = 4
N_ROUNDS = 6
N_INITIAL = 12
TOTAL_NEW_EVALUATIONS = BATCH_SIZE * N_ROUNDS
```

These should not be editable in the competition notebook unless the instructor explicitly changes them.

### Student-editable object

Students should edit only:

```python
TEAM_STRATEGY = [
    {"nehvi": 4},
    {"nehvi": 3, "random": 1},
    {"parego": 4},
    {"nehvi": 2, "scalarized_0.5_0.5": 2},
    {"nehvi": 4},
    {"nehvi": 4},
]
```

Optionally expose projection method:

```python
PROJECTION_METHOD = "diverse_nearest"
```

Avoid exposing too many free-form hooks.

### Competition metrics

Primary metric:

```text
AUC-HV over rounds
```

Tie-breakers:

1. Final hypervolume.
2. Number of non-dominated selected sequences.
3. Diversity of selected VH embeddings.
4. Earliest round reaching a target HV.

### Leaderboard output

Each run should write:

```text
outputs/
  team_name_run_id.json
  team_name_history.csv
  team_name_pareto_plot.png
  team_name_hv_curve.png
```

Suggested JSON contents:

```json
{
  "team_name": "...",
  "seed": 123,
  "batch_size": 4,
  "n_rounds": 6,
  "auc_hv": 0.0,
  "final_hv": 0.0,
  "selected_ids": [],
  "strategy": [],
  "projection_method": "diverse_nearest"
}
```

### Anti-confusion rules

The competition notebook should validate:

1. Fixed batch size every round.
2. No duplicate selected sequence IDs.
3. No evaluating already-observed IDs.
4. No direct access to hidden full objective table.
5. No changing the oracle.
6. No changing initial design.
7. No changing budget.

### Final conclusions section: true Pareto-front reveal

At the end of the competition notebook or in a short instructor-led final notebook section, reveal the hidden true objective landscape.

The goal is to turn the leaderboard into a scientific discussion rather than only a ranking exercise.

The final reveal should include:

1. Scatter plot of the full candidate library in true objective space.
2. The true Pareto front highlighted.
3. Each student group's achieved non-dominated front overlaid.
4. Optionally, each group's selected dominated points shown faintly.
5. The initial design highlighted separately.
6. Final HV and AUC-HV displayed for each team.

Suggested function:

```python
plot_true_front_with_team_overlays(
    Y_true_all=oracle.true_objectives,
    initial_ids=initial_ids,
    team_runs=loaded_team_runs,
    ref_point=REF_POINT,
)
```

The plot should make it possible to ask:

```text
Which parts of the true Pareto front were discovered?
Which parts were missed?
Did different strategies discover different regions?
Did scalarization over-focus on one part of the frontier?
Did qLogNEHVI expand already-known Pareto regions efficiently?
Did qLogNParEGO or mixed strategies cover more diverse trade-offs?
Did exploration help find isolated Pareto regions?
```

Recommended visual layers:

```text
gray points: all candidate sequences
black curve/points: true Pareto front
large outlined points: initial design
colored points/lines: achieved Pareto front for each team
small faint colored points: all selected points for each team
reference point: marked explicitly
```

Discussion prompts:

1. Where did fixed scalarization perform well?
   - Usually near the region matching its chosen preference weights.
2. Where did fixed scalarization struggle?
   - Usually in frontier regions misaligned with the fixed weights.
3. Where did qLogNEHVI perform well?
   - Usually in regions where the model already had enough information to estimate likely hypervolume gains.
4. Where might qLogNEHVI miss?
   - Potentially isolated frontier regions that require exploration before their hypervolume contribution is visible.
5. Where did qLogNParEGO help?
   - Potentially by sampling different preference directions and spreading search across the frontier.
6. When did random or uncertainty-driven slots help?
   - When they discovered latent regions not yet represented in the observed data.
7. What did the projection step change?
   - Continuous optima may map to repeated or nearby sequences; diversity-aware projection can affect frontier coverage.

Coding-agent tasks:

1. Add a hidden or instructor-only `oracle.true_objectives` access path.
2. Ensure students cannot call true objectives during the competition phase.
3. Implement `compute_true_pareto_front(Y_true_all)`.
4. Implement `plot_true_front_with_team_overlays(...)`.
5. Implement `load_team_runs(output_dir)`.
6. Add a final debrief cell that loads all team submissions and produces the overlay plot.
7. Export the final reveal plot to `outputs/final_true_pareto_overlay.png`.
8. Add a short Markdown prompt cell asking students to interpret which acquisition strategies had advantages in which regions of objective space.


---

## 11. Notebook 04: Optional extensions

This notebook is for faster or more advanced students.

Possible extensions:

### 11.1 Custom scalarization schedule

Students design a round-dependent scalarization schedule:

```python
weights_by_round = [
    [0.5, 0.5],
    [0.8, 0.2],
    [0.2, 0.8],
    [0.5, 0.5],
]
```

### 11.2 Exploration-to-exploitation schedule

Start with mixed exploration, then switch to qLogNEHVI:

```python
TEAM_STRATEGY = [
    {"random": 2, "parego": 2},
    {"parego": 4},
    {"nehvi": 2, "parego": 2},
    {"nehvi": 4},
    {"nehvi": 4},
    {"nehvi": 4},
]
```

### 11.3 Alternative projection rules

Compare:

1. Nearest latent neighbor.
2. Diverse nearest neighbor.
3. Cluster-aware selection.
4. Sequence-similarity penalty.

### 11.4 Information-theoretic acquisitions

Optional conceptual discussion only, unless time permits.

If implemented, it should be heavily scaffolded. Students should not be expected to debug these methods in the main lab.

---

## 12. Helper module architecture

Recommended package structure:

```text
mobo_lab/
  __init__.py
  config.py
  seed.py
  data.py
  pool.py
  oracle.py
  models.py
  acquisitions.py
  optimize.py
  projection.py
  metrics.py
  plotting.py
  strategies.py
  competition.py
  verification.py
```

### 12.1 `config.py`

Define shared constants:

```python
SEED = 123
BATCH_SIZE = 4
N_INITIAL = 12
N_ROUNDS = 6
LATENT_DIM = 5
NUM_OBJECTIVES = 2
NUM_RESTARTS = 10
RAW_SAMPLES = 128
MC_SAMPLES = 64
REF_POINT = [-0.05, -0.05]
```

### 12.2 `seed.py`

```python
def set_all_seeds(seed: int) -> None:
    ...
```

### 12.3 `data.py`

Load:

1. VH sequence CSV.
2. Latent embeddings.
3. Initial IDs.
4. Optional metadata.

Expected API:

```python
def load_sequence_table(path): ...
def load_latents(path): ...
def load_initial_ids(path): ...
```

### 12.4 `pool.py`

Class:

```python
class VHSequencePool:
    X: torch.Tensor
    sequences: list[str]
    ids: list[int]

    def available_ids(self, observed_ids, pending_ids=None): ...
    def project_to_unqueried_sequences(self, candidates, observed_ids, pending_ids=None, method="nearest"): ...
```

### 12.5 `oracle.py`

Class:

```python
class AntibodyOracle:
    def evaluate(self, ids: list[int]) -> torch.Tensor:
        ...
```

Implementation choices:

1. Use a hidden synthetic function over embeddings.
2. Use precomputed hidden objective arrays.
3. Add deterministic seeded noise per `(sequence_id, replicate)` or per evaluation call.

For reproducibility in the golden-path notebook, prefer deterministic pseudo-noise keyed by sequence ID and global seed.

### 12.6 `models.py`

Expose:

```python
def fit_surrogate_model(train_X, train_Y, train_Yvar=None):
    ...
```

Internal implementation can use independent single-task GPs combined into a model list. Keep details hidden from students unless they inspect the helper.

### 12.7 `acquisitions.py`

Expose:

```python
def build_acquisition(
    name: str,
    model,
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    ref_point: torch.Tensor,
    sampler,
    **kwargs,
):
    ...
```

Required names:

```python
"nehvi"
"parego"
"scalarized_0.5_0.5"
"scalarized_0.8_0.2"
"scalarized_0.2_0.8"
"random"
"uncertainty"
```

### 12.8 `optimize.py`

Expose:

```python
def optimize_continuous_acquisition(
    acq_func,
    bounds,
    q: int,
    num_restarts: int,
    raw_samples: int,
    sequential: bool = True,
):
    ...
```

This wraps `optimize_acqf`. The default should be `sequential=True` for all core notebooks.

### 12.9 `strategies.py`

Expose:

```python
def validate_batch_plan(batch_plan: dict[str, int], batch_size: int) -> None:
    ...

def propose_batch_from_plan(
    batch_plan,
    model,
    pool,
    observed_ids,
    train_X,
    train_Y,
    ref_point,
    bounds,
    batch_size,
    projection_method,
):
    ...
```

The function should handle mixed plans such as:

```python
{"nehvi": 2, "parego": 2}
```

Implementation detail:

- For each strategy card, generate the requested number of continuous candidates using sequential greedy optimization.
- Project to available sequence IDs while tracking pending IDs.
- Condition later greedy selections on earlier pending selections whenever possible.
- Combine into a single batch of length `BATCH_SIZE`.

### 12.10 `metrics.py`

Expose:

```python
def compute_pareto_mask(Y): ...
def compute_true_pareto_front(Y_true_all): ...
def compute_hypervolume(Y, ref_point): ...
def compute_auc_hv(hv_history): ...
def compute_embedding_diversity(X_selected): ...
```

### 12.11 `plotting.py`

Expose:

```python
def plot_pareto_front(Y, selected_mask=None, ref_point=None, title=None): ...
def plot_hv_curve(hv_history, title=None): ...
def plot_true_front_with_team_overlays(
    Y_true_all,
    initial_ids,
    team_runs,
    ref_point,
    output_path=None,
): ...
```

### 12.12 `competition.py`

Expose:

```python
def run_campaign(team_strategy, team_name, seed, projection_method): ...
def save_run_outputs(history, output_dir): ...
def load_team_runs(output_dir): ...
def update_leaderboard(output_dir): ...
def build_final_debrief_report(output_dir, oracle, initial_ids, ref_point): ...
```

### 12.13 `verification.py`

Expose:

```python
def verify_golden_path(candidate_ids, new_Y, hv_before, hv_after): ...
```

---

## 13. Data files

### 13.0 Source data and parsing (status: done, 2026-06-27)

Raw input: `data/vh_data.xlsx` (single sheet, 113 rows). Columns: `Sample ID`,
`Seq`, `Germline`, `Germline_Identity%`, `Tm (oC)`, `BV Elisa score`,
`Affinity, Kd (nM)`, `Yield (mg per 10 mL culture)`.

- `scripts/parse_data.py` cleans this into `data/vh_data.csv` (snake_case names,
  numeric `germline_identity_pct`, derived `length`; validates unique IDs and
  standard amino acids). Tested in `tests/scripts/test_parse_data.py`.
- `scripts/visualize_data.py` writes EDA figures to `docs/figures/`
  (distributions, correlations, trade-off views, sequence overview).

Library facts to keep in mind for the rest of the build:

- 113 unique sequences, lengths 112-130, 20 standard amino acids only.
- 43 distinct IGHV germline genes; the library is diverse but **small**. 113 may
  be too few for a rich competition pool (cf. §18.3) -- expanding the sequence
  library is a likely follow-up.
- Latent embeddings (`vh_latents.npy`) and oracle objective files are still TODO;
  they will be derived from the sequences, not from the raw spreadsheet.

Recommended structure:

```text
data/
  vh_sequences.csv
  vh_latents.npy
  initial_indices.json
  oracle_observed_objectives.npy
  oracle_true_objectives.npy       # optional hidden file
  metadata.json
```

### 13.1 `vh_sequences.csv`

Columns:

```text
sequence_id,sequence,length,optional_family,optional_notes
```

### 13.2 `vh_latents.npy`

Shape:

```text
[num_sequences, latent_dim]
```

Values should be scaled to `[0, 1]`.

### 13.3 `initial_indices.json`

Example:

```json
{
  "seed": 123,
  "initial_ids": [1, 7, 12, 31, 44, 58, 79, 103, 145, 201, 233, 377]
}
```

### 13.4 Oracle objective files

If using precomputed oracle values:

```text
oracle_observed_objectives.npy: [num_sequences, 2]
oracle_true_objectives.npy: [num_sequences, 2]
```

If using deterministic pseudo-noisy observations, store only true objectives and generate noisy observations through the oracle.

---

## 14. Reproducibility requirements

For Notebook 01, reproducibility is essential.

The coding agent should:

1. Fix Python, NumPy, and PyTorch seeds.
2. Use CPU unless GPU reproducibility is confirmed.
3. Use deterministic initial IDs.
4. Use deterministic oracle noise.
5. Record package versions.
6. Include expected candidate IDs, objective values, and HV values.
7. Provide a `verify_golden_path` cell.

Potential issue:

`optimize_acqf` may produce slightly different continuous candidates across environments. Because candidates are projected to nearest sequence IDs, small numeric changes may or may not affect projected IDs.

Mitigations:

1. Use CPU.
2. Use low-dimensional latent space.
3. Use well-separated candidate embeddings.
4. Use deterministic restart initialization if possible.
5. Make the expected check compare projected IDs and HV, not exact continuous latent candidates.
6. If necessary, use a small fixed candidate set for Notebook 01 acquisition optimization, then use `optimize_acqf` with `sequential=True` in Notebook 02/03.

---

## 15. Student-facing narrative

Use this storyline:

```text
We are optimizing antibodies under limited experimental throughput.
Each round, the wet lab can test exactly four candidates.
Each candidate has two properties we care about.
The properties may trade off, so there is no single best antibody.
Instead, we want to discover a set of strong trade-offs: the Pareto front.
```

Then introduce acquisition functions as strategy choices:

```text
qLogNEHVI asks: which batch is expected to expand the Pareto front the most?
qLogNParEGO asks: what if we repeatedly sample different preferences over the two objectives?
Fixed scalarization asks: what if we decide ahead of time how much we care about each objective?
Random/exploration asks: what if we spend part of the wet-lab budget learning uncertain regions?
```

Avoid saying students need to understand GP training. Instead:

```text
The surrogate model gives us predictions and uncertainty. Today we will focus on
how to use that uncertainty to choose the next batch of experiments.
```

---

## 16. Instructor notes

### 16.1 What to explain live

Explain:

1. Why multiple objectives create trade-offs.
2. Why a fixed batch size corresponds to wet-lab throughput.
3. Why hypervolume is a useful single-number score.
4. Why noisy observations motivate noisy acquisition functions.
5. Why continuous latent optimization needs projection back to valid sequences.
6. Why different acquisition strategies behave differently.

Do not spend much time on:

1. GP kernels.
2. Marginal likelihood.
3. Detailed derivation of EHVI.
4. Sequence autoencoder training.
5. Information-theoretic acquisitions.

### 16.2 Common student confusions

Prepare explanations for:

1. `q` is the number of parallel candidates, not the number of objectives.
2. MC samples are not the batch size.
3. `X_baseline` means already observed design points.
4. The reference point should be worse than meaningful objective values.
5. BoTorch assumes maximization by default in this setup.
6. Continuous latent candidates are not automatically valid sequences.
7. A batch acquisition can be optimized jointly or sequentially. In this practicum, we use sequential greedy batch construction.

---

## 17. Minimal viable implementation plan

If time is limited, implement in this order:

1. Prepare `data/` with VH sequences and latents.
2. Implement oracle with deterministic noisy two-objective outputs.
3. Implement Pareto/HV utilities and plotting.
4. Implement `fit_surrogate_model`.
5. Implement qLogNEHVI acquisition + `optimize_acqf` with `sequential=True`.
6. Implement candidate projection.
7. Build Notebook 01 golden path.
8. Add qLogNParEGO.
9. Add fixed scalarization.
10. Build strategy-card abstraction.
11. Build competition notebook and leaderboard.

Notebook 01 should be completed before adding competition features.

---


## 18. Instructor-only preflight: calibrating competition difficulty

This section is for the instructor/developer, not for the student-facing notebooks.

Before finalizing the practicum, run several candidate student strategies offline to make sure the task is rich enough for competition-based exploration.

### 18.1 Run strategy sweeps

Evaluate the default strategy cards under the same initial design, budget, and batch size:

```python
strategies_to_test = {
    "all_nehvi": [
        {"nehvi": 4},
        {"nehvi": 4},
        {"nehvi": 4},
        {"nehvi": 4},
        {"nehvi": 4},
        {"nehvi": 4},
    ],
    "all_parego": [
        {"parego": 4},
        {"parego": 4},
        {"parego": 4},
        {"parego": 4},
        {"parego": 4},
        {"parego": 4},
    ],
    "scalarization_sweep": [
        {"scalarized_0.8_0.2": 2, "scalarized_0.2_0.8": 2},
        {"scalarized_0.8_0.2": 2, "scalarized_0.2_0.8": 2},
        {"scalarized_0.5_0.5": 4},
        {"scalarized_0.5_0.5": 4},
        {"nehvi": 4},
        {"nehvi": 4},
    ],
    "explore_then_exploit": [
        {"random": 2, "parego": 2},
        {"parego": 4},
        {"nehvi": 2, "parego": 2},
        {"nehvi": 4},
        {"nehvi": 4},
        {"nehvi": 4},
    ],
    "mixed": [
        {"nehvi": 2, "parego": 2},
        {"nehvi": 3, "random": 1},
        {"scalarized_0.8_0.2": 2, "scalarized_0.2_0.8": 2},
        {"nehvi": 2, "parego": 2},
        {"nehvi": 4},
        {"nehvi": 4},
    ],
}
```

For each strategy, record:

1. AUC-HV.
2. Final HV.
3. HV improvement per round.
4. Number of non-dominated selected sequences.
5. Diversity of selected embeddings.
6. Whether selected points cover multiple regions of the Pareto front.

### 18.2 Desired difficulty properties

The task should not be so easy that all strategies recover the Pareto front quickly. It should also not be so hard or noisy that all strategies look random.

A good competition instance should have:

1. Clear but nontrivial trade-offs between the two objectives.
2. A Pareto front that is not fully discovered by the initial design.
3. Multiple useful regions of the Pareto front, so fixed scalarization can over-focus.
4. Enough latent structure that model-guided strategies beat random on average.
5. Enough uncertainty/noise that exploration can be useful.
6. Enough sequence diversity that projection does not collapse many candidates to near-duplicates.
7. A visible difference between all-NEHVI, ParEGO, scalarization, random, and mixed strategies.

### 18.3 Iterate on sequence set and oracle

If the strategy sweeps are not interesting, iterate on the sequence subset and/or oracle function.

Possible adjustments:

1. Change the VH sequence subset:
   - increase or decrease library size,
   - choose a more diverse subset in latent space,
   - remove near-duplicates,
   - ensure several distinct latent clusters contain Pareto-relevant candidates.

2. Change the latent dimension:
   - too low may make the task too simple,
   - too high may make acquisition optimization and modeling unstable for students.

3. Change the hidden oracle:
   - tune the strength of the trade-off between objectives,
   - add or remove local optima,
   - adjust noise level,
   - add motif-based bonuses or penalties,
   - make one Pareto region easier to find than another,
   - ensure objective values remain normalized and interpretable.

4. Change the initial design:
   - avoid seeding too many Pareto-optimal points,
   - include enough diversity to fit a reasonable initial surrogate,
   - use the same initial design for all teams.

5. Change budget:
   - if all methods saturate quickly, reduce `N_ROUNDS` or increase difficulty,
   - if no methods improve, increase `N_INITIAL`, increase `N_ROUNDS`, or simplify the oracle.

### 18.4 Preflight acceptance criteria

Before teaching, the instructor should be able to verify:

1. The golden-path notebook reproduces exactly.
2. qLogNEHVI usually beats random in AUC-HV.
3. At least one mixed strategy can sometimes beat all-NEHVI.
4. Fixed scalarization visibly concentrates in one part of objective space.
5. ParEGO explores different trade-offs across rounds.
6. The final leaderboard is not predetermined by a single obvious strategy.
7. Runtime fits comfortably within the practicum schedule.
8. Plots are visually interpretable for students.
9. The true Pareto-front reveal produces a meaningful contrast between achieved fronts and missed regions.


## 19. Open decisions

These can be changed later.

1. Exact VH sequence dataset.
2. Whether objectives are real labels or synthetic oracle values.
3. Latent representation method.
4. Latent dimension.
5. Whether oracle noise is deterministic per sequence or per evaluation.
6. Whether leaderboard uses noisy observed HV or hidden true HV.
7. Whether to include full joint batch optimization as an optional comparison. Core notebooks should use sequential greedy optimization.
8. Whether qLogNParEGO uses direct `qLogNParEGO` or an explicit list of scalarized noisy EI acquisitions.
9. Whether diversity projection is mandatory or optional.
10. Whether students submit strategies live or through saved JSON files.

---

## 20. Suggested defaults

Recommended first pass:

```python
LATENT_DIM = 5
NUM_OBJECTIVES = 2
N_INITIAL = 12
BATCH_SIZE = 4
N_ROUNDS = 6
MC_SAMPLES = 64
NUM_RESTARTS = 10
RAW_SAMPLES = 128
REF_POINT = [-0.05, -0.05]
PROJECTION_METHOD = "nearest"
PRIMARY_SCORE = "auc_hv"
TIE_BREAKER = "final_hv"
```

Required strategies:

```python
"nehvi"
"parego"
"scalarized_0.5_0.5"
"scalarized_0.8_0.2"
"scalarized_0.2_0.8"
"random"
```

Optional strategies:

```python
"uncertainty"
"diverse_projection"
"custom_weights"
```

---

## 21. Desired student takeaways

By the end of the practicum, students should be able to say:

1. Multi-objective optimization searches for trade-offs, not one universal best point.
2. Pareto fronts describe non-dominated trade-offs.
3. Hypervolume summarizes Pareto-front quality relative to a reference point.
4. Batch BO selects multiple experiments at once, matching wet-lab throughput.
5. Sequential greedy batch construction fills the fixed batch one candidate at a time.
6. Noisy acquisition functions account for uncertainty in observed outcomes.
7. BoTorch acquisition functions can be optimized with `optimize_acqf`.
8. In latent biological design, continuous optimized candidates must be mapped back to valid designs.
9. qLogNEHVI, qLogNParEGO, and fixed scalarization represent different campaign strategies.
10. Comparing achieved fronts to the true Pareto front helps diagnose which acquisition functions are strong in which objective-space regions.
