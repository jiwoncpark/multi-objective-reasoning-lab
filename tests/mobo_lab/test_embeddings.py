"""Tests for ``mobo_lab/embeddings.py`` (descriptor featurizer + PCA latent builder)."""

from __future__ import annotations

import numpy as np
import pytest

from mobo_lab import config, embeddings


def _toy_sequences() -> list[str]:
    """Eight short, compositionally distinct sequences (well separated in latent space)."""
    return [
        "ACDEFGHIKLMNPQRSTVWY",
        "AAAACCCCDDDDEEEEFFFF",
        "KLKLKLKLKLKLKLKLKLKL",
        "MNPQRSTVWYMNPQRSTVWY",
        "GGGGGGSSSSSSAAAAAAYY",
        "WYWYWYFWFWFWYHYHYHYH",
        "DEDEDEKRKRKRDEKRDEKR",
        "VVVVIIIILLLLMMMMFFFF",
    ]


def test_descriptor_features_shape_and_composition_sums_to_one():
    feats = embeddings.descriptor_features(_toy_sequences())
    assert feats.shape == (8, 20 + embeddings.NUM_AGGREGATE_FEATURES)
    composition = feats[:, :20]  # the 20 amino-acid fractions
    np.testing.assert_allclose(composition.sum(axis=1), 1.0, atol=1e-9)


def test_descriptor_features_known_counts():
    feats = embeddings.descriptor_features(["AACC"])  # 2 A, 2 C, length 4
    assert feats[0, embeddings.AA_ORDER.index("A")] == 0.5
    assert feats[0, embeddings.AA_ORDER.index("C")] == 0.5
    # aromatic fraction is 0 here; normalized length is 4 / LENGTH_SCALE
    assert feats[0, 22] == 0.0
    assert feats[0, 23] == 4.0 / embeddings.LENGTH_SCALE


def test_build_latents_shape_and_range():
    latents = embeddings.build_latents(_toy_sequences())
    assert latents.shape == (8, config.LATENT_DIM)
    assert latents.min() >= 0.0
    assert latents.max() <= 1.0
    # min-max should make every retained component span the full unit interval
    np.testing.assert_allclose(latents.min(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(latents.max(axis=0), 1.0, atol=1e-9)


def test_build_latents_deterministic():
    a = embeddings.build_latents(_toy_sequences())
    b = embeddings.build_latents(_toy_sequences())
    np.testing.assert_array_equal(a, b)  # byte-identical across calls


def test_sign_fix_invariant_to_component_sign_flip():
    rng = np.random.default_rng(0)
    components = rng.standard_normal((config.LATENT_DIM, 24))
    fixed = embeddings._sign_fix(components)

    flipped = components.copy()
    flipped[1] *= -1
    flipped[3] *= -1
    # flipping any component's sign must not change the sign-fixed result
    np.testing.assert_array_equal(embeddings._sign_fix(flipped), fixed)

    # and each fixed component's largest-magnitude loading is positive
    for row in fixed:
        assert row[int(np.argmax(np.abs(row)))] > 0


def test_min_pairwise_distance_simple():
    pts = np.array([[0.0, 0.0], [0.0, 3.0], [4.0, 0.0]])
    dist, i, j = embeddings.min_pairwise_distance(pts)
    assert dist == 3.0
    assert {i, j} == {0, 1}


def test_unknown_backend_raises():
    with pytest.raises(KeyError):
        embeddings.build_latents(["ACDE"], backend="does_not_exist")
