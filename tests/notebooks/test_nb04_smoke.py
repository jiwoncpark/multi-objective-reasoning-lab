"""Smoke test: Notebook 04 (optional extensions) executes headless.

The notebook runs several full campaigns (custom scalarization schedule,
explore->exploit, nearest vs diverse_nearest, an uncertainty card) and prints
comparison tables. A clean top-to-bottom run is the acceptance check. It is the
slowest notebook (multiple campaigns, including the continuous path), hence the
generous timeout. mobo_lab is installed editable.
"""

from __future__ import annotations

from pathlib import Path

import jupytext
import pytest
from nbclient import NotebookClient

NB_PY = Path(__file__).resolve().parents[2] / "notebooks" / "04_optional_extensions.py"


@pytest.mark.slow
def test_notebook04_executes():
    assert NB_PY.exists(), f"missing notebook source: {NB_PY}"
    nb = jupytext.read(NB_PY)
    client = NotebookClient(nb, timeout=900, kernel_name="python3")
    client.execute()  # raises CellExecutionError on any failing cell
