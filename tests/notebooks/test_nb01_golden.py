"""Golden-path test: Notebook 01 executes headless and reproduces the frozen run.

The notebook's own ``verify_golden_path`` cell raises on any drift, so a clean
top-to-bottom execution *is* the reproducibility guarantee. We also assert the
frozen constants are present, so re-freezing the golden path can't silently empty
them.
"""

from __future__ import annotations

from pathlib import Path

import jupytext
import pytest
from nbclient import NotebookClient

from mobo_lab import verification

NB_PY = (
    Path(__file__).resolve().parents[2]
    / "notebooks"
    / "01_seeded_noisy_sequential_greedy_mobo_iteration.py"
)


def test_frozen_constants_present():
    assert verification.EXPECTED_CANDIDATE_IDS, "golden constants must be frozen (Step 11)"
    assert len(verification.EXPECTED_NEW_Y) == len(verification.EXPECTED_CANDIDATE_IDS)


@pytest.mark.slow
def test_notebook01_executes_and_verifies():
    assert NB_PY.exists(), f"missing notebook source: {NB_PY}"
    nb = jupytext.read(NB_PY)
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()  # the verify_golden_path cell raises on any mismatch

    # Confirm the success banner actually printed (guards against the cell being removed).
    printed = "".join(
        out.get("text", "")
        for cell in nb.cells
        if cell.cell_type == "code"
        for out in cell.get("outputs", [])
    )
    assert "Golden-path check passed" in printed
    assert str(verification.EXPECTED_CANDIDATE_IDS) in printed
