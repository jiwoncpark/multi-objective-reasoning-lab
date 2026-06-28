"""Tests for ``mobo_lab/verification.py`` (golden-path self-check).

Uses placeholder expected constants injected via ``monkeypatch``; when Step 11
freezes the real golden path, only those numbers change -- this test structure
stays valid.
"""

from __future__ import annotations

import pytest

from mobo_lab import verification
from mobo_lab.verification import verify_golden_path

# A stand-in "frozen" golden path for the tests.
_IDS = [3, 7, 11, 19]
_NEW_Y = [[0.4, 0.6], [0.5, 0.5], [0.6, 0.4], [0.3, 0.7]]
_HV_BEFORE = 0.40
_HV_AFTER = 0.65


@pytest.fixture(autouse=True)
def _frozen(monkeypatch):
    monkeypatch.setattr(verification, "EXPECTED_CANDIDATE_IDS", _IDS)
    monkeypatch.setattr(verification, "EXPECTED_NEW_Y", _NEW_Y)
    monkeypatch.setattr(verification, "EXPECTED_HV_BEFORE", _HV_BEFORE)
    monkeypatch.setattr(verification, "EXPECTED_HV_AFTER", _HV_AFTER)


def test_passes_on_matching_values(capsys):
    verify_golden_path(_IDS, _NEW_Y, _HV_BEFORE, _HV_AFTER)
    assert "Golden-path check passed" in capsys.readouterr().out


def test_raises_on_mismatched_ids():
    with pytest.raises(AssertionError, match="candidate IDs"):
        verify_golden_path([3, 7, 11, 20], _NEW_Y, _HV_BEFORE, _HV_AFTER)


def test_raises_on_too_different_new_Y():
    bad = [[0.4, 0.6], [0.5, 0.5], [0.6, 0.4], [0.3, 0.99]]
    with pytest.raises(AssertionError):
        verify_golden_path(_IDS, bad, _HV_BEFORE, _HV_AFTER)


def test_accepts_tiny_new_Y_noise():
    jittered = [[v + 1e-7 for v in row] for row in _NEW_Y]
    verify_golden_path(_IDS, jittered, _HV_BEFORE, _HV_AFTER)  # within tolerance


def test_raises_when_hv_decreases(monkeypatch):
    # Pin the expected HVs to a (hypothetical) decreasing pair so the values still
    # "match" expectations but trip the monotonicity guard.
    monkeypatch.setattr(verification, "EXPECTED_HV_BEFORE", 0.65)
    monkeypatch.setattr(verification, "EXPECTED_HV_AFTER", 0.40)
    with pytest.raises(AssertionError, match="decreased"):
        verify_golden_path(_IDS, _NEW_Y, hv_before=0.65, hv_after=0.40)


def test_raises_on_mismatched_hv():
    with pytest.raises(AssertionError, match="hv_after"):
        verify_golden_path(_IDS, _NEW_Y, _HV_BEFORE, 0.99)


def test_raises_when_constants_not_frozen(monkeypatch):
    monkeypatch.setattr(verification, "EXPECTED_CANDIDATE_IDS", [])
    with pytest.raises(RuntimeError, match="not frozen yet"):
        verify_golden_path(_IDS, _NEW_Y, _HV_BEFORE, _HV_AFTER)
