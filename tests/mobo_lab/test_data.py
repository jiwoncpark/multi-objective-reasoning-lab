"""Tests for ``mobo_lab/data.py`` loaders, using tiny temp files."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from mobo_lab import config, data


def _write_library(tmp_path, rows):
    path = tmp_path / "vh_library.csv"
    pd.DataFrame(rows, columns=data.LIBRARY_COLUMNS).to_csv(path, index=False)
    return path


def _good_rows():
    return [
        {"sequence_id": "VH-1", "sequence": "QVQLVQSG", "length": 8, "source": "u1", "cluster_id": 0},
        {"sequence_id": "VH-2", "sequence": "EVQLLESG", "length": 8, "source": "u1", "cluster_id": 1},
    ]


def test_load_library_happy_path(tmp_path):
    path = _write_library(tmp_path, _good_rows())
    df = data.load_library(path)
    assert list(df.columns) == data.LIBRARY_COLUMNS
    assert len(df) == 2
    assert data.load_sequences(path) == ["QVQLVQSG", "EVQLLESG"]


def test_load_library_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_library(tmp_path / "nope.csv")


def test_load_library_duplicate_id_raises(tmp_path):
    rows = _good_rows()
    rows[1]["sequence_id"] = "VH-1"
    with pytest.raises(ValueError, match="duplicate"):
        data.load_library(_write_library(tmp_path, rows))


def test_load_library_bad_residue_raises(tmp_path):
    rows = _good_rows()
    rows[0]["sequence"] = "QVQLXVSG"  # X is not a standard residue
    with pytest.raises(ValueError, match="non-standard"):
        data.load_library(_write_library(tmp_path, rows))


def test_load_library_missing_column_raises(tmp_path):
    path = tmp_path / "vh_library.csv"
    pd.DataFrame([{"sequence_id": "VH-1", "sequence": "QVQL"}]).to_csv(path, index=False)
    with pytest.raises(KeyError):
        data.load_library(path)


def test_load_latents_ok_and_validation(tmp_path):
    path = tmp_path / "vh_latents.npy"
    np.save(path, np.full((5, config.LATENT_DIM), 0.5))
    latents = data.load_latents(path)
    assert latents.shape == (5, config.LATENT_DIM)
    assert latents.dtype == torch.double

    np.save(path, np.full((5, config.LATENT_DIM + 1), 0.5))
    with pytest.raises(ValueError, match="shape"):
        data.load_latents(path)

    np.save(path, np.full((5, config.LATENT_DIM), 1.5))  # out of [0, 1]
    with pytest.raises(ValueError, match="0, 1"):
        data.load_latents(path)


def test_load_initial_ids(tmp_path):
    path = tmp_path / "initial_indices.json"
    path.write_text(json.dumps({"seed": 123, "initial_ids": [1, 7, 12]}))
    assert data.load_initial_ids(path) == [1, 7, 12]


def test_load_true_objectives(tmp_path):
    path = tmp_path / "oracle_true_objectives.npy"
    np.save(path, np.zeros((4, config.NUM_OBJECTIVES)))
    obj = data.load_true_objectives(path)
    assert obj.shape == (4, config.NUM_OBJECTIVES)

    np.save(path, np.zeros((4, 3)))
    with pytest.raises(ValueError, match="shape"):
        data.load_true_objectives(path)
