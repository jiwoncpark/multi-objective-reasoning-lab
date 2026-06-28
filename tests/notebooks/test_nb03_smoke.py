"""Smoke test: Notebook 03 (competition) runs a full campaign + reveal headless.

A clean top-to-bottom run exercises run_campaign, save_run_outputs,
update_leaderboard, and the instructor true-front reveal. The run_campaign call
itself enforces the §10 anti-confusion rules, so completion is the acceptance
check. Writes go to the gitignored ``outputs/`` dir.
"""

from __future__ import annotations

from pathlib import Path

import jupytext
import pytest
from nbclient import NotebookClient

NB_PY = Path(__file__).resolve().parents[2] / "notebooks" / "03_competition.py"


@pytest.mark.slow
def test_notebook03_executes():
    assert NB_PY.exists(), f"missing notebook source: {NB_PY}"
    nb = jupytext.read(NB_PY)
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()  # raises CellExecutionError if the campaign or reveal fails
