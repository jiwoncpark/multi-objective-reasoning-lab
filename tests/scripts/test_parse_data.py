"""Tests for ``scripts/parse_data.py``.

The tests build tiny in-memory spreadsheets with hand-checkable numbers, so the
expected values are obvious by inspection. We cover:

* the happy path (clean rename, derived length, parsed percentage),
* the percentage parser on string and already-numeric input,
* every validation guard (missing column, duplicate ID, bad amino acid),
* a smoke test against the real ``data/vh_data.xlsx`` if it is present.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Make ``scripts/`` importable without installing it as a package.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import parse_data  # noqa: E402
from parse_data import (  # noqa: E402
    OUTPUT_COLUMNS,
    _parse_percent,
    parse_vh_data,
    summarize,
)


def _write_xlsx(tmp_path: Path, rows: list[dict]) -> Path:
    """Write rows (using the *raw* spreadsheet headers) to a temp .xlsx."""
    path = tmp_path / "mini.xlsx"
    pd.DataFrame(rows).to_excel(path, index=False)
    return path


# A minimal but realistic two-row fixture using the raw header names.
GOOD_ROWS = [
    {
        "Sample ID": "AB-1",
        "Seq": "QVQLVQSG",  # 8 residues, all standard
        "Germline": "IGHV1-46*01",
        "Germline_Identity%": "77.1%",
        "Tm (oC)": 49.5,
        "BV Elisa score": 0.32,
        "Affinity, Kd (nM)": 37.8,
        "Yield (mg per 10 mL culture)": 0.24,
    },
    {
        "Sample ID": "AB-2",
        "Seq": "evklves",  # lower-case, 7 residues -> should be upper-cased
        "Germline": "IGHV1-24*01",
        "Germline_Identity%": "90.6%",
        "Tm (oC)": 56.3,
        "BV Elisa score": 0.20,
        "Affinity, Kd (nM)": 140.0,
        "Yield (mg per 10 mL culture)": 0.60,
    },
]


def test_parse_happy_path(tmp_path: Path) -> None:
    df = parse_vh_data(_write_xlsx(tmp_path, GOOD_ROWS))

    # Columns and order match the published contract.
    assert list(df.columns) == OUTPUT_COLUMNS
    assert len(df) == 2

    # Sequences are upper-cased and stripped.
    assert df.loc[1, "sequence"] == "EVKLVES"

    # Length is derived correctly.
    assert df.loc[0, "length"] == 8
    assert df.loc[1, "length"] == 7

    # Percentage is parsed to a float in percent units.
    assert df.loc[0, "germline_identity_pct"] == pytest.approx(77.1)
    assert pd.api.types.is_float_dtype(df["germline_identity_pct"])

    # Biophysical numbers pass through unchanged.
    assert df.loc[0, "tm_celsius"] == pytest.approx(49.5)
    assert df.loc[1, "affinity_kd_nm"] == pytest.approx(140.0)


def test_parse_percent_handles_strings_and_numbers() -> None:
    as_str = pd.Series(["50%", "12.5%", "100%"])
    out = _parse_percent(as_str)
    assert out.tolist() == pytest.approx([50.0, 12.5, 100.0])

    # Already-numeric input is passed through as float.
    as_num = pd.Series([50, 12.5, 100])
    assert _parse_percent(as_num).tolist() == pytest.approx([50.0, 12.5, 100.0])


def test_missing_column_raises(tmp_path: Path) -> None:
    rows = [dict(GOOD_ROWS[0])]
    del rows[0]["Tm (oC)"]
    with pytest.raises(KeyError, match="Tm"):
        parse_vh_data(_write_xlsx(tmp_path, rows))


def test_duplicate_sample_id_raises(tmp_path: Path) -> None:
    rows = [dict(GOOD_ROWS[0]), dict(GOOD_ROWS[1])]
    rows[1]["Sample ID"] = "AB-1"  # force a collision
    with pytest.raises(ValueError, match="Duplicate sample IDs"):
        parse_vh_data(_write_xlsx(tmp_path, rows))


def test_non_standard_amino_acid_raises(tmp_path: Path) -> None:
    rows = [dict(GOOD_ROWS[0])]
    rows[0]["Seq"] = "QVQLXZ"  # X and Z are not standard amino acids
    with pytest.raises(ValueError, match="non-standard amino-acid"):
        parse_vh_data(_write_xlsx(tmp_path, rows))


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        parse_vh_data("does/not/exist.xlsx")


def test_summarize_mentions_counts(tmp_path: Path) -> None:
    df = parse_vh_data(_write_xlsx(tmp_path, GOOD_ROWS))
    text = summarize(df)
    assert "Parsed 2 antibody VH sequences." in text
    assert "tm_celsius" in text


@pytest.mark.skipif(
    not parse_data.DEFAULT_INPUT.exists(),
    reason="real data/vh_data.xlsx not present",
)
def test_real_dataset_smoke() -> None:
    df = parse_vh_data()
    assert len(df) == 113
    assert list(df.columns) == OUTPUT_COLUMNS
    assert df["sample_id"].is_unique
    # Every sequence is a clean standard-amino-acid string.
    assert df["length"].min() >= 1
    # Percentages are in a sane 0-100 range.
    assert df["germline_identity_pct"].between(0, 100).all()
