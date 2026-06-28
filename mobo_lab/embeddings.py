"""Sequence -> latent design-vector embeddings for the candidate library.

Each curated VH amino-acid string is turned into a continuous design vector
``x in [0, 1]^LATENT_DIM`` that BoTorch then optimizes over a standard ``[0, 1]``
box. Because the oracle's two objectives are a *synthetic* function of these
latents (see ``docs/07``), the embedding does **not** need to predict any real
biophysical property -- it only has to keep the ``N`` sequences well separated and
reasonably smooth in the latent cube.

Design goals:

* **Deterministic, CPU-only, no downloads** by default. The ``descriptor_pca``
  backend turns each string into transparent amino-acid descriptors, reduces them
  with an *exact* SVD-based PCA, and pins a fixed sign convention, so two runs
  produce byte-identical latents on any BLAS build.
* **Pluggable backend.** A pretrained antibody language model
  (``igbert_unpaired``) can be swapped in later behind the same interface. Doing
  so changes the latent *geometry*, so the oracle and the frozen golden-path
  constants would have to be regenerated (Steps 6, 10, 11).

The reduction pipeline shared by all backends is::

    raw features [N, F] -> z-score columns -> exact PCA (top k, sign-fixed)
                        -> min-max each component to [0, 1]
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from . import config

# Canonical amino-acid ordering used for the composition features.
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"

# Kyte & Doolittle hydropathy scale (higher = more hydrophobic).
KD_HYDROPATHY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "V": 4.2, "Y": -1.3,
}

# Net-charge contribution at pH ~7 (His carries a small partial positive charge).
CHARGE = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1}

# Aromatic side chains.
AROMATIC = frozenset("FWY")

# Length normalizer (curated lengths are ~90-150; the exact value is irrelevant
# because the column is z-scored before PCA -- it only keeps the raw feature
# numerically comparable to the fractions).
LENGTH_SCALE = 100.0

# Number of aggregate descriptors appended after the 20 composition fractions.
NUM_AGGREGATE_FEATURES = 4

# A backend maps a list of sequences to a raw ``[N, F]`` feature matrix.
EmbeddingBackend = Callable[[list[str]], np.ndarray]


# --------------------------------------------------------------------------- #
# Backends: sequences -> raw features [N, F]
# --------------------------------------------------------------------------- #
def descriptor_features(sequences: list[str]) -> np.ndarray:
    """Deterministic amino-acid descriptors: ``[N, 24]``.

    Columns are, in order:

    * ``0..19``  -- composition fractions for ``AA_ORDER`` (sum to 1 per row when
      every residue is standard),
    * ``20``     -- mean Kyte-Doolittle hydrophobicity,
    * ``21``     -- net charge per residue at pH ~7,
    * ``22``     -- aromatic fraction,
    * ``23``     -- length normalized by ``LENGTH_SCALE``.
    """
    n_aa = len(AA_ORDER)
    aa_index = {aa: j for j, aa in enumerate(AA_ORDER)}
    feats = np.zeros((len(sequences), n_aa + NUM_AGGREGATE_FEATURES), dtype=np.float64)
    for i, seq in enumerate(sequences):
        length = len(seq)
        denom = length or 1  # guard the (unexpected) empty sequence
        kd = charge = aromatic = 0.0
        for ch in seq:
            j = aa_index.get(ch)
            if j is None:  # non-standard residue: ignored (load_library forbids these)
                continue
            feats[i, j] += 1.0
            kd += KD_HYDROPATHY[ch]
            charge += CHARGE.get(ch, 0.0)
            if ch in AROMATIC:
                aromatic += 1.0
        feats[i, :n_aa] /= denom
        feats[i, n_aa + 0] = kd / denom
        feats[i, n_aa + 1] = charge / denom
        feats[i, n_aa + 2] = aromatic / denom
        feats[i, n_aa + 3] = length / LENGTH_SCALE
    return feats


def igbert_features(sequences: list[str]) -> np.ndarray:
    """Mean-pooled residue embeddings from a pretrained antibody LM: ``[N, hidden]``.

    Optional backend. Lazy-imports ``transformers`` and downloads
    ``Exscientia/IgBert_unpaired`` on first use (instructor/GPU side), so it is not
    exercised by the offline test suite. The output is fed through the *same*
    standardize -> PCA -> sign-fix -> min-max pipeline as the default backend.
    """
    import torch  # lazy: keep the default path import-light
    from transformers import AutoModel, AutoTokenizer

    model_name = "Exscientia/IgBert_unpaired"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()

    spaced = [" ".join(seq) for seq in sequences]  # IgBert expects space-separated residues
    pooled: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(spaced), 32):
            batch = spaced[start : start + 32]
            enc = tokenizer(batch, return_tensors="pt", padding=True)
            hidden = model(**enc).last_hidden_state.double()  # [b, L, H]
            mask = enc["attention_mask"].unsqueeze(-1).double()
            mean = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            pooled.append(mean.cpu().numpy())
    return np.concatenate(pooled, axis=0)


BACKENDS: dict[str, EmbeddingBackend] = {
    "descriptor_pca": descriptor_features,
    "igbert_unpaired": igbert_features,
}


# --------------------------------------------------------------------------- #
# Shared reduction pipeline: raw features -> [N, k] in [0, 1]
# --------------------------------------------------------------------------- #
def standardize_columns(features: np.ndarray) -> np.ndarray:
    """Z-score each column; constant columns (std 0) are left centered at 0."""
    mu = features.mean(axis=0)
    sd = features.std(axis=0)
    sd = np.where(sd == 0.0, 1.0, sd)
    return (features - mu) / sd


def _sign_fix(components: np.ndarray) -> np.ndarray:
    """Pin each component's sign so its largest-magnitude loading is positive.

    Returns a fixed *copy*. This removes the global sign ambiguity that SVD
    leaves on each singular vector (different BLAS builds may return a component
    or its negation), making the resulting scores byte-stable across machines.
    """
    components = np.array(components, dtype=np.float64, copy=True)
    for i in range(components.shape[0]):
        j = int(np.argmax(np.abs(components[i])))
        if components[i, j] < 0:
            components[i] = -components[i]
    return components


def pca_sign_fixed(features_std: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact PCA via ``numpy.linalg.svd`` with deterministic component signs.

    ``features_std`` must already be column-centered (``standardize_columns``).
    Returns ``(scores [N, n_components], explained_variance_ratio [n_components])``.
    If fewer than ``n_components`` non-trivial directions exist, the trailing
    score columns and ratios are zero-padded.
    """
    _u, singular_values, vt = np.linalg.svd(features_std, full_matrices=False)
    k = min(n_components, vt.shape[0])
    components = _sign_fix(vt[:k])  # [k, F] loadings, signs pinned
    scores = features_std @ components.T  # [N, k]

    variance = singular_values**2
    total = variance.sum()
    ratio = variance[:k] / total if total > 0 else np.zeros(k)

    if k < n_components:  # tiny inputs: pad to the requested width with zeros
        pad = n_components - k
        scores = np.concatenate([scores, np.zeros((scores.shape[0], pad))], axis=1)
        ratio = np.concatenate([ratio, np.zeros(pad)])
    return scores, ratio


def minmax_unit(scores: np.ndarray) -> np.ndarray:
    """Min-max each column into ``[0, 1]``; constant columns map to all-zeros."""
    lo = scores.min(axis=0)
    span = scores.max(axis=0) - lo
    span = np.where(span == 0.0, 1.0, span)  # constant column -> (x-lo)/1 == 0
    return (scores - lo) / span


def build_latents_with_info(
    sequences: list[str],
    backend: str = "descriptor_pca",
    n_components: int = config.LATENT_DIM,
    seed: int = config.SEED,
) -> tuple[np.ndarray, dict]:
    """Build latents and return ``(latents [N, k] in [0, 1], info)``.

    ``info`` carries ``explained_variance_ratio`` and ``n_features`` for the build
    summary. ``seed`` is accepted for interface symmetry; the default backend is
    fully deterministic and does not use it.
    """
    if backend not in BACKENDS:
        raise KeyError(f"unknown backend {backend!r}; choices: {sorted(BACKENDS)}")
    sequences = list(sequences)
    features = np.asarray(BACKENDS[backend](sequences), dtype=np.float64)
    if features.ndim != 2 or features.shape[0] != len(sequences):
        raise ValueError(
            f"backend {backend!r} returned features of shape {features.shape}, "
            f"expected [{len(sequences)}, F]"
        )
    scores, ratio = pca_sign_fixed(standardize_columns(features), n_components)
    latents = minmax_unit(scores)
    info = {"explained_variance_ratio": ratio, "n_features": int(features.shape[1])}
    return latents, info


def build_latents(
    sequences: list[str],
    backend: str = "descriptor_pca",
    n_components: int = config.LATENT_DIM,
    seed: int = config.SEED,
) -> np.ndarray:
    """Build the ``[N, n_components]`` latent design matrix in ``[0, 1]``."""
    latents, _info = build_latents_with_info(sequences, backend, n_components, seed)
    return latents


def min_pairwise_distance(points: np.ndarray) -> tuple[float, int, int]:
    """Return ``(distance, i, j)`` for the closest pair of rows (Euclidean).

    ``(inf, -1, -1)`` when fewer than two points are given. Uses the
    ``||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b`` identity so the full pairwise check
    over a few thousand points stays a couple of cheap matrix ops.
    """
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n < 2:
        return float("inf"), -1, -1
    gram = points @ points.T
    sq = np.diag(gram)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * gram, 0.0)
    np.fill_diagonal(d2, np.inf)
    flat = int(np.argmin(d2))
    i, j = divmod(flat, n)
    return float(np.sqrt(d2[i, j])), i, j
