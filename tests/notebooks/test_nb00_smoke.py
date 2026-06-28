"""Smoke test: Notebook 00 executes top-to-bottom on the python3 kernel.

We read the jupytext percent source (the diffable source of truth) and run it with
nbclient. This catches notebook rot whenever the helper API changes. mobo_lab is
installed editable, so imports resolve regardless of the execution directory.
"""

from __future__ import annotations

from pathlib import Path

import jupytext
import pytest
from nbclient import NotebookClient

NB_PY = Path(__file__).resolve().parents[2] / "notebooks" / "00_pareto_hypervolume_warmup.py"


@pytest.mark.slow
def test_notebook00_executes():
    assert NB_PY.exists(), f"missing notebook source: {NB_PY}"
    nb = jupytext.read(NB_PY)
    client = NotebookClient(nb, timeout=300, kernel_name="python3")
    client.execute()  # raises CellExecutionError on any failing cell
