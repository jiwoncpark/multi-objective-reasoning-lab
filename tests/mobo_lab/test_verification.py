"""Tests for ``mobo_lab/verification.py`` (golden-path self-check).

These assert against the **real frozen constants** (Step 11). If the golden path
is ever re-frozen, only the module constants change and these tests still hold.
"""

from __future__ import annotations

import pytest
import torch

from mobo_lab import verification
from mobo_lab.verification import verify_golden_path


def test_constants_are_frozen():
    # Step 11 froze real values; guard against an accidental reset to placeholders.
    assert len(verification.EXPECTED_CANDIDATE_IDS) == 4
    assert len(verification.EXPECTED_NEW_Y) == 4
    assert verification.EXPECTED_HV_AFTER >= verification.EXPECTED_HV_BEFORE > 0.0


def test_passes_on_the_frozen_values(capsys):
    verify_golden_path(
        verification.EXPECTED_CANDIDATE_IDS,
        verification.EXPECTED_NEW_Y,
        verification.EXPECTED_HV_BEFORE,
        verification.EXPECTED_HV_AFTER,
    )
    assert "Golden-path check passed" in capsys.readouterr().out


def test_accepts_tiny_new_Y_noise():
    jittered = [[v + 1e-7 for v in row] for row in verification.EXPECTED_NEW_Y]
    verify_golden_path(
        verification.EXPECTED_CANDIDATE_IDS,
        jittered,
        verification.EXPECTED_HV_BEFORE,
        verification.EXPECTED_HV_AFTER,
    )  # within tolerance, no raise


def test_raises_on_mismatched_ids():
    bad_ids = list(verification.EXPECTED_CANDIDATE_IDS)
    bad_ids[0] += 1
    with pytest.raises(AssertionError, match="candidate IDs"):
        verify_golden_path(
            bad_ids,
            verification.EXPECTED_NEW_Y,
            verification.EXPECTED_HV_BEFORE,
            verification.EXPECTED_HV_AFTER,
        )


def test_raises_on_too_different_new_Y():
    bad_Y = [list(row) for row in verification.EXPECTED_NEW_Y]
    bad_Y[0][0] += 0.5
    with pytest.raises(AssertionError):
        verify_golden_path(
            verification.EXPECTED_CANDIDATE_IDS,
            bad_Y,
            verification.EXPECTED_HV_BEFORE,
            verification.EXPECTED_HV_AFTER,
        )


def test_raises_on_mismatched_hv():
    with pytest.raises(AssertionError, match="hv_after"):
        verify_golden_path(
            verification.EXPECTED_CANDIDATE_IDS,
            verification.EXPECTED_NEW_Y,
            verification.EXPECTED_HV_BEFORE,
            verification.EXPECTED_HV_AFTER + 0.5,
        )


def test_raises_when_hv_decreases(monkeypatch):
    # Pin the expected HVs to a (hypothetical) decreasing pair so the values still
    # "match" expectations but trip the monotonicity guard.
    monkeypatch.setattr(verification, "EXPECTED_HV_BEFORE", 0.65)
    monkeypatch.setattr(verification, "EXPECTED_HV_AFTER", 0.40)
    with pytest.raises(AssertionError, match="decreased"):
        verify_golden_path(
            verification.EXPECTED_CANDIDATE_IDS, verification.EXPECTED_NEW_Y, 0.65, 0.40
        )


def test_raises_when_constants_not_frozen(monkeypatch):
    monkeypatch.setattr(verification, "EXPECTED_CANDIDATE_IDS", [])
    with pytest.raises(RuntimeError, match="not frozen yet"):
        verify_golden_path([1, 2, 3, 4], verification.EXPECTED_NEW_Y, 0.1, 0.2)
