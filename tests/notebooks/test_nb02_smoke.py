"""Smoke test: Notebook 02 (strategy-card practice) executes top-to-bottom.

The notebook itself asserts, for every plan, that the batch is BATCH_SIZE distinct
unqueried IDs and that hypervolume is non-decreasing -- so a clean headless run is
the acceptance check. mobo_lab is installed editable, so imports resolve anywhere.
"""

from __future__ import annotations

from pathlib import Path

import jupytext
import pytest
from nbclient import NotebookClient

NB_PY = Path(__file__).resolve().parents[2] / "notebooks" / "02_strategy_cards_practice.py"


@pytest.mark.slow
def test_notebook02_executes():
    assert NB_PY.exists(), f"missing notebook source: {NB_PY}"
    nb = jupytext.read(NB_PY)
    client = NotebookClient(nb, timeout=300, kernel_name="python3")
    client.execute()  # raises CellExecutionError if any plan's invariants fail
