# Step 5 — Latent embeddings (`vh_latents.npy`)

**Status:** DONE (2026-06-27)
**Depends on:** Step 4 (`vh_library.csv`, `data.py`)
**Unblocks:** oracle (Step 6, defined over latents), pool/projection (Step 7), the BO engine (Step 8).

**Result:** `mobo_lab/embeddings.py` + `scripts/build_latents.py` materialize
`data/vh_latents.npy` `[2048, 5] ⊂ [0,1]`, byte-identical across runs (verified by
`cmp`). The default `descriptor_pca` backend's top-5 PCA components explain
0.154 / 0.107 / 0.094 / 0.074 / 0.069 of variance (sum ≈ 0.50); the min pairwise
latent distance is **0.0166** (above `τ = 1e-2`), so no two sequences collapse for
the nearest-neighbour projection. 11 unit/end-to-end tests green.

## Goal

Turn each curated VH sequence into a continuous design vector `x ∈ [0,1]^5` so BoTorch can optimize
acquisition functions over a standard box. The embedding must be **deterministic, CPU-only, no
downloads** by default, with a clean hook to later swap in a pretrained antibody language model
(`IgBert_unpaired`). Because the oracle is defined over these latents, the embedding does **not** need
to predict any real property — it only needs to keep the `N` sequences well separated and reasonably
smooth.

## Files to create

```text
scripts/build_latents.py            # backend-pluggable: sequences -> data/vh_latents.npy
mobo_lab/embeddings.py              # backend registry + featurizers (importable, testable)
tests/scripts/test_build_latents.py
tests/mobo_lab/test_embeddings.py
```

## Backend interface (`mobo_lab/embeddings.py`)

```python
EmbeddingBackend = Callable[[list[str]], np.ndarray]   # sequences -> [N, F] raw features

def descriptor_features(sequences: list[str]) -> np.ndarray:    # default backend, F ~ 24
    ...
def igbert_features(sequences: list[str]) -> np.ndarray:        # optional; lazy-imports transformers
    ...

BACKENDS = {"descriptor_pca": descriptor_features, "igbert_unpaired": igbert_features}

def build_latents(sequences, backend="descriptor_pca", n_components=config.LATENT_DIM,
                  seed=config.SEED) -> np.ndarray:               # -> [N, n_components] in [0,1]
    ...
```

### Default backend: `descriptor_pca`

1. **Featurize (deterministic from the string):** 20 amino-acid composition fractions + a few
   aggregate descriptors — mean Kyte–Doolittle hydrophobicity, net charge at pH 7, aromatic fraction,
   normalized length (~24 features total).
2. **Standardize** columns (z-score).
3. **Exact PCA via `numpy.linalg.svd`** (deterministic — **not** randomized PCA). Keep top
   `LATENT_DIM=5` components.
4. **Sign-fix** each component deterministically (e.g. force the loading of largest magnitude to be
   positive) so `vh_latents.npy` is byte-identical across BLAS builds.
5. **Min–max scale** each component to `[0,1]` so `bounds = [0,1]^5` is exact.

### Optional backend: `igbert_unpaired`

- Lazy-import `transformers`; load `Exscientia/IgBert_unpaired` (or equivalent), mean-pool residue
  embeddings → `[N, hidden]`, then the **same** standardize → SVD → sign-fix → min-max pipeline down
  to 5 components. Gated behind the backend flag; needs a one-time model download (instructor/GPU).
- Swapping backends changes the latent geometry ⇒ **regenerate oracle + golden values** (Steps 6, 10,
  11). Document this coupling in the script header.

## `scripts/build_latents.py`

CLI: `python scripts/build_latents.py --backend descriptor_pca --out data/vh_latents.npy`

- Load sequences via `data.load_sequences`, call `embeddings.build_latents`, save `np.save`.
- **Post-build guards (fail loudly):** shape `[N, 5]`; values in `[0,1]`; **minimum pairwise Euclidean
  distance `> τ`** (e.g. `τ ≈ 1e-2`) so no two sequences collapse to near-duplicate latents
  (protects the projection step, outline §18.2.6). If a pair is too close, report the offending IDs
  (instructor then drops/curates one upstream in Step 4).
- Print a summary: per-dimension min/max/spread, min pairwise distance, explained-variance ratio.

## Implementation notes

- Determinism is paramount: no randomized SVD, fixed sign convention, no shuffling. Two runs ⇒
  identical bytes (tested).
- 2048 points in 5-D is ample separation for distinct sequences; the curation in Step 4 already
  removed isolated modes, so PCA yields a fairly uniform cloud (min pairwise distance 0.0166).
- The default backend is intentionally simple and transparent; its purpose is geometry, not biology
  (the oracle supplies the "biology" synthetically).

## Tests

- `test_embeddings.py`: `descriptor_features` composition row sums to ~1 over the 20 AA fractions;
  `build_latents` output shape `[N,5]`, range `⊂[0,1]`, identical across two calls (determinism);
  `_sign_fix` is invariant to flipping any component's sign (the SVD ambiguity it removes) and pins
  each component's largest-magnitude loading positive; `min_pairwise_distance` returns the closest
  pair on a hand-checked example.

  > Note: the original draft proposed testing invariance to flipping an *input feature*'s sign. That
  > only holds when the flipped feature is not the largest-magnitude loading (anchor) of a retained
  > component, so the sound, unconditional guarantee — invariance to flipping a *component*'s sign,
  > which is the actual cross-BLAS ambiguity — is tested instead.
- `test_build_latents.py`: end-to-end on a small sequence list → npy with correct shape/range; the
  min-pairwise-distance guard raises on a deliberately duplicated sequence.

## Acceptance criteria

- `uv run python scripts/build_latents.py` writes `data/vh_latents.npy` `[N,5] ⊂ [0,1]`, min pairwise
  distance above τ, reproducibly.
- `uv run pytest tests/scripts/test_build_latents.py tests/mobo_lab/test_embeddings.py` green.
