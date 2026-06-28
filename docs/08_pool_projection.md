# Step 7 — Sequence pool + continuous→discrete projection

**Status:** DONE (2026-06-28)
**Depends on:** Step 5 (`vh_latents.npy`), Step 4 (`data.py`)
**Unblocks:** strategies (Step 9), notebooks 01–03, the competition closed loop.

**Result:** `mobo_lab/projection.py` (low-level `nearest` / `diverse_nearest` +
`METHODS` dispatch) and `mobo_lab/pool.py` (`VHSequencePool`) implement the
continuous→discrete bridge. `VHSequencePool.from_files()` loads the real assets:
`X` is `[2048, 5] ⊂ [0,1]` double, `ids == range(2048)`, aligned with `sequences`.
Both projection strategies are greedy and order-preserving, take a `forbidden` set
they never mutate (private copy), and raise `ValueError` on pool exhaustion. The
diversity term weight is a module constant `DIVERSITY_WEIGHT = 1.0`; for the first
pick (empty within-batch chosen set) `diverse_nearest` reduces exactly to
`nearest`. Identity holds: exact pool rows project back to their own IDs (the
property that makes the discrete golden path's projection exact). End-to-end check
on the real pool: a `[4, 5]` proposal batch projects to 4 distinct unqueried IDs,
`diverse_nearest` returns a more spread set than `nearest`, and `oracle.evaluate`
accepts the IDs directly. 22 new tests green (120 total).

> **Design note:** the doc-specified low-level signature is
> `diverse_nearest(candidates, pool_X, forbidden)`, so the diversity repulsion is
> from the **within-call chosen batch only** — `pending_ids` passed to
> `project_to_unqueried_sequences` contribute to *exclusion* (the `forbidden` set)
> but not to the repulsion anchor. The golden path projects the whole `q`-batch in
> a single call with no pending, so this is exactly within-batch diversity.
> Cross-call pending-aware diversity is a deferred extension.

## Goal

Wrap the finite candidate library as a `VHSequencePool` and implement the projection that turns a
continuous acquisition proposal `z* ∈ [0,1]^5` into a **valid, unqueried** sequence ID. This is the
"continuous latent candidate → nearest valid sequence → ID → oracle" bridge (outline §3.3).

Note: with the **discrete** graded golden path (Step 11), projection is no longer load-bearing for
reproducibility — candidates returned by `optimize_acqf_discrete` are already pool rows, so projection
is an identity lookup. Projection still matters for the **continuous** path taught in the syntax cell
and used as an option in Notebook 02/03.

## Files to create

```text
mobo_lab/pool.py
mobo_lab/projection.py
tests/mobo_lab/test_pool.py
tests/mobo_lab/test_projection.py
```

## `mobo_lab/projection.py` — low-level helpers

```python
def nearest(candidates: Tensor, pool_X: Tensor, forbidden: set[int]) -> list[int]:
    """For each candidate row, the index of the nearest pool row (L2) not in `forbidden`,
    updating `forbidden` as it goes so a batch never repeats an ID."""

def diverse_nearest(candidates: Tensor, pool_X: Tensor, forbidden: set[int]) -> list[int]:
    """Like `nearest`, but penalizes proximity to already-chosen pending points so the batch
    spreads out (outline §3.3 method='diverse_nearest')."""

METHODS = {"nearest": nearest, "diverse_nearest": diverse_nearest}
```

## `mobo_lab/pool.py`

```python
class VHSequencePool:
    X: Tensor            # [N, LATENT_DIM] double, in [0,1]
    sequences: list[str]
    ids: list[int]       # 0..N-1 (row index == sequence row)

    @classmethod
    def from_files(cls, library_csv=config.LIBRARY_CSV, latents_npy=config.LATENTS_NPY): ...

    def available_ids(self, observed_ids, pending_ids=None) -> list[int]: ...

    def project_to_unqueried_sequences(self, candidates: Tensor, observed_ids,
                                       pending_ids=None, method="nearest") -> list[int]:
        """[q, d] continuous candidates -> q distinct, unqueried sequence IDs."""
```

`project_to_unqueried_sequences` builds `forbidden = set(observed_ids) | set(pending_ids or [])`,
dispatches to `projection.METHODS[method]`, and returns `q` distinct IDs, none observed/pending.
Handles the §3.3 edge cases: already-observed IDs excluded; duplicate continuous candidates mapping to
the same row resolved to distinct rows; graceful fallback when the nearest row is already taken (take
the next-nearest available).

## Implementation notes

- Use plain `torch.cdist` for pairwise distances; `N≈256`, `q=4` → trivial cost.
- IDs are integer row indices into `X`/`sequences` (so `pool.X[ids]` indexes directly, matching outline
  §8.3 `train_X = pool.X[initial_ids]`).
- `available_ids` returns the complement of observed∪pending, used by the `"random"` strategy and by
  `optimize_acqf_discrete`'s `X_avoid` construction.
- For an exact pool row as input, `nearest` returns that row's ID (identity lookup) — this is what
  makes the discrete golden path's projection exact.

## Tests

Use a tiny hand-laid pool, e.g. `X = [[0,0],[0,1],[1,0],[1,1],[0.5,0.5]]` (d=2 for readability):

- `nearest`: a candidate at `[0.1,0.1]` → row 0; with `forbidden={0}` → next-nearest (row 4).
- Projecting `q` candidates never returns observed or pending IDs; returns `q` **distinct** IDs.
- Two identical candidates project to two **different** rows (no within-batch duplicate).
- `diverse_nearest` vs `nearest`: for clustered candidates, `diverse_nearest` selects a more spread set
  (assert the chosen set's min pairwise distance is ≥ that of `nearest`).
- Identity: feeding exact pool rows returns those rows' IDs in order.
- `available_ids` excludes observed∪pending and has the right length.

## Acceptance criteria

- `uv run pytest tests/mobo_lab/test_pool.py tests/mobo_lab/test_projection.py` green.
- `VHSequencePool.from_files()` loads, asserts `X.shape == (N, LATENT_DIM)`, `0 ≤ X ≤ 1`.
