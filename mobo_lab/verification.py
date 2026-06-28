"""Golden-path self-check for the seeded competition loop (Notebook 01).

After students run the first deterministic round, :func:`verify_golden_path`
confirms their results match the instructor-frozen reference, so a single failing
assertion points at the exact step that drifted instead of a silently wrong
campaign. It compares the **projected IDs, observed objectives, and hypervolumes**
only -- never the continuous acquisition candidates, which are not reproducible
across machines (outline §14 mitigation #5); the reproducibility guarantee runs
through the discrete-pool path.

The ``EXPECTED_*`` constants are placeholders here and are **frozen in Step 11**
once the golden path is finalized. Until then :func:`verify_golden_path` raises a
clear "not frozen yet" error rather than comparing against the placeholders.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

# --- Frozen in Step 11 after the golden path is finalized -------------------- #
EXPECTED_CANDIDATE_IDS: list[int] = []
EXPECTED_NEW_Y: list[list[float]] = []
EXPECTED_HV_BEFORE: float = 0.0
EXPECTED_HV_AFTER: float = 0.0

# Tolerances for the floating-point comparisons (outline §8.10).
_Y_RTOL = 1e-5
_Y_ATOL = 1e-6
_HV_ATOL = 1e-6

_SUCCESS_MESSAGE = "Golden-path check passed. You are ready for the strategy-card notebook."


def verify_golden_path(
    candidate_ids: Iterable[int],
    new_Y,
    hv_before: float,
    hv_after: float,
) -> None:
    """Assert the round-1 results match the frozen golden path.

    Checks, in order: the candidate IDs equal :data:`EXPECTED_CANDIDATE_IDS`; the
    observations ``new_Y`` are close to :data:`EXPECTED_NEW_Y`; the hypervolumes
    match :data:`EXPECTED_HV_BEFORE` / :data:`EXPECTED_HV_AFTER`; and the
    hypervolume did not go down. Prints a success line and returns ``None`` on
    pass; raises ``AssertionError`` on the first mismatch.

    Raises ``RuntimeError`` if the expected constants have not been frozen yet
    (Step 11).
    """
    if not EXPECTED_CANDIDATE_IDS:
        raise RuntimeError(
            "golden-path expected constants are not frozen yet; they are filled "
            "in Step 11 once the golden path is finalized."
        )

    ids = [int(i) for i in candidate_ids]
    if ids != list(EXPECTED_CANDIDATE_IDS):
        raise AssertionError(
            f"candidate IDs {ids} != expected {list(EXPECTED_CANDIDATE_IDS)}"
        )

    new_Y = torch.as_tensor(new_Y, dtype=torch.double)
    expected_Y = torch.tensor(EXPECTED_NEW_Y, dtype=torch.double)
    torch.testing.assert_close(new_Y, expected_Y, rtol=_Y_RTOL, atol=_Y_ATOL)

    if abs(float(hv_before) - EXPECTED_HV_BEFORE) > _HV_ATOL:
        raise AssertionError(
            f"hv_before {hv_before} != expected {EXPECTED_HV_BEFORE}"
        )
    if abs(float(hv_after) - EXPECTED_HV_AFTER) > _HV_ATOL:
        raise AssertionError(f"hv_after {hv_after} != expected {EXPECTED_HV_AFTER}")
    if float(hv_after) < float(hv_before):
        raise AssertionError(
            f"hypervolume decreased: hv_after {hv_after} < hv_before {hv_before}"
        )

    print(_SUCCESS_MESSAGE)
