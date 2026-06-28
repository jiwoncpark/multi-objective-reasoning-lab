# Step 1 — Package foundations: `config` + `seed`

**Status:** DONE (2026-06-27) — `mobo_lab/{__init__,config,seed}.py` + tests; 9/9 green, full suite 21/21.
**Depends on:** nothing
**Unblocks:** every other step (all modules import `config`; reproducibility relies on `seed`).

> **Implementation notes (as built):** `pyproject.toml` was made an installable package
> (`[build-system]` setuptools + `[tool.setuptools] packages = ["mobo_lab"]`) so notebooks can
> `import mobo_lab`; run `uv sync` once to install it into the env. Tests resolve the package via
> `[tool.pytest.ini_options] pythonpath = ["."]` with `--import-mode=importlib`, so they pass with or
> without the install. `seed.py` uses a relative import (`from . import config`).

## Goal

Create the `mobo_lab` package skeleton and the two leaf modules every other module and notebook
depends on: a single source of truth for constants (`config.py`) and a one-call reproducibility
helper (`seed.py`). No BoTorch yet.

## Files to create

```text
mobo_lab/__init__.py
mobo_lab/config.py
mobo_lab/seed.py
tests/mobo_lab/__init__.py            # if needed for discovery
tests/mobo_lab/test_config.py
tests/mobo_lab/test_seed.py
```

Also add `mobo_lab` to the package list in `pyproject.toml` (`[tool.setuptools] packages`/`uv`
project config) so `from mobo_lab import config` resolves when running notebooks via `uv run`.

## Public API

### `config.py` — module-level constants (authoritative; outline §12.1, §20)

```python
SEED = 123

# Campaign geometry
BATCH_SIZE = 4
N_INITIAL = 12
N_ROUNDS = 6
TOTAL_NEW_EVALUATIONS = BATCH_SIZE * N_ROUNDS   # 24

# Spaces
LATENT_DIM = 5                 # NOTE: authoritative value; outline §3.2's "8" is stale
NUM_OBJECTIVES = 2
LIBRARY_SIZE = 2048            # 2^11 curated pool size (Step 4): NN library + ground-truth front

# Acquisition / optimizer
NUM_RESTARTS = 10
RAW_SAMPLES = 128
MC_SAMPLES = 64
REF_POINT = [-0.05, -0.05]     # length == NUM_OBJECTIVES; worse-than-any-real-objective

# Oracle noise (per objective std; Step 6)
NOISE_SIGMA = (0.05, 0.05)

# Defaults
PROJECTION_METHOD = "nearest"
PRIMARY_SCORE = "auc_hv"
TIE_BREAKER = "final_hv"

# Filesystem (resolve relative to repo root, not CWD)
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUTS_DIR = REPO_ROOT / "outputs"
LIBRARY_CSV = DATA_DIR / "vh_library.csv"
LATENTS_NPY = DATA_DIR / "vh_latents.npy"
INITIAL_IDS_JSON = DATA_DIR / "initial_indices.json"
ORACLE_TRUE_NPY = DATA_DIR / "oracle_true_objectives.npy"
```

Provide a small helper so notebooks can grab a torch `REF_POINT`/`bounds` without re-deriving:

```python
def ref_point_tensor(dtype=torch.double) -> torch.Tensor: ...
def latent_bounds(dtype=torch.double) -> torch.Tensor:     # shape [2, LATENT_DIM], rows = (0, 1)
    ...
```

### `seed.py`

```python
def set_all_seeds(seed: int = config.SEED) -> None:
    """Seed random, numpy, and torch (CPU) for reproducible runs."""
    # random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    # torch.set_default_dtype(torch.double); device stays CPU
```

## Implementation notes

- Keep `config.py` import-light: only `pathlib` + `torch` (for the tensor helpers). Do **not** import
  BoTorch here (keeps the leaf cheap and avoids the qLogNEHVI fused-kernel import cost at config time).
- `set_all_seeds` should also `torch.set_default_dtype(torch.double)` and leave device on CPU, matching
  the golden-path requirement (outline §8 "Use CPU for deterministic behavior").
- Do **not** call `torch.use_deterministic_algorithms(True)` by default (it can error on some ops);
  document it as optional (outline §8.1).
- These constants are duplicated as visible literals inside notebooks for pedagogy (outline §8.3);
  `config.py` is the importable source of truth for the library/competition code. Keep them in sync.

## Tests (`tests/mobo_lab/`)

- `test_config.py`: `len(REF_POINT) == NUM_OBJECTIVES`; `LATENT_DIM == 5`;
  `TOTAL_NEW_EVALUATIONS == BATCH_SIZE * N_ROUNDS`; `latent_bounds().shape == (2, LATENT_DIM)` with
  row0 all-zeros, row1 all-ones; paths are absolute and rooted at the repo.
- `test_seed.py`: after `set_all_seeds(0)`, drawing `torch.rand(3)` / `np.random.rand(3)` /
  `random.random()` twice (re-seeding between) yields identical sequences; default dtype is float64.

## Acceptance criteria

- `uv run python -c "from mobo_lab import config, seed; seed.set_all_seeds(); print(config.LATENT_DIM)"`
  prints `5`.
- `uv run pytest tests/mobo_lab/test_config.py tests/mobo_lab/test_seed.py` is green.
