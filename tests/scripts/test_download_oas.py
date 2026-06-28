"""Offline tests for ``scripts/download_oas.py`` (HF streaming design).

We feed synthetic row dicts (the shape ``datasets`` yields) straight into the pure
filter functions -- no network and no ``datasets`` import (the loader is separate
and imports ``datasets`` lazily).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import download_oas as dl  # noqa: E402


def _seq(n: int, start: str = "") -> str:
    body = "ACDEFGHIKLMNPQRSTVWY" * (n // 20 + 1)
    return (start + body)[:n]


def _distinct(i: int) -> str:
    """A valid length-100 sequence, unique per ``i`` (differs at position ``i``)."""
    s = list(_seq(100))
    s[i] = "C" if s[i] != "C" else "A"
    return "".join(s)


def _row(**over) -> dict:
    base = {
        "meta_Species": "human",
        "meta_Chain": "Heavy",
        "meta_BSource": "PBMC",
        "meta_BType": "Memory-B-Cells",
        "meta_Isotype": "IGHG",
        "meta_Run": "RUN1",
        "meta_Author": "Author",
        "productive": "T",
        "vj_in_frame": "T",
        "stop_codon": "F",
        "ANARCI_status": "",
        "sequence_alignment_aa": _seq(100),
        "locus": "IGH",
    }
    base.update(over)
    return base


# --------------------------------------------------------------------------- #
def test_to_bool():
    assert dl.to_bool("T") and dl.to_bool("true") and dl.to_bool("1") and dl.to_bool(True)
    assert not dl.to_bool("F") and not dl.to_bool("false") and not dl.to_bool("") and not dl.to_bool(None)


def test_clean_vh_aa_and_only_standard():
    assert dl.clean_vh_aa("qv.ql-vq sg") == "QVQLVQSG"
    assert dl.clean_vh_aa(None) == ""
    assert dl._only_standard("QVQL")
    assert not dl._only_standard("QVXL")
    assert not dl._only_standard("")


def test_row_passes_good():
    ok, cleaned = dl.row_passes(_row())
    assert ok and len(cleaned) == 100


def test_is_valid_heavy():
    # ANARCII result shape: chain_type/score/error per sequence.
    good = {"chain_type": "H", "score": 29.8, "error": None}
    light = {"chain_type": "K", "score": 25.0, "error": None}
    failed = {"chain_type": "F", "score": 0.0, "error": "Less than 50 ... numbered."}
    assert dl._is_valid_heavy(good)
    assert not dl._is_valid_heavy(light)  # not a heavy chain
    assert not dl._is_valid_heavy(failed)  # numbering error
    assert not dl._is_valid_heavy(good, min_score=50.0)  # below score floor


def test_row_passes_strips_gaps():
    raw = _seq(100)
    ok, cleaned = dl.row_passes(_row(sequence_alignment_aa=raw[:40] + "..." + raw[40:]))
    assert ok and cleaned == raw


@pytest.mark.parametrize(
    "over",
    [
        {"meta_Species": "mouse"},
        {"meta_BType": "Plasma-B-Cells"},
        {"meta_Isotype": "IGHE"},
        {"meta_Chain": "Light"},
        {"productive": "F"},
        {"vj_in_frame": "F"},
        {"stop_codon": "T"},
        {"sequence_alignment_aa": ""},
        {"sequence_alignment_aa": _seq(50)},   # too short
        {"sequence_alignment_aa": _seq(200)},  # too long
        {"sequence_alignment_aa": _seq(99) + "X"},  # ambiguous residue
    ],
)
def test_row_passes_rejects(over):
    ok, _ = dl.row_passes(_row(**over))
    assert not ok


def test_collect_filtered_stream_dedup():
    g1, g2 = _seq(100, "A"), _seq(100, "K")
    rows = [
        _row(sequence_alignment_aa=g1),
        _row(sequence_alignment_aa=g1),  # exact duplicate
        _row(sequence_alignment_aa=g2),
        _row(meta_Species="mouse", sequence_alignment_aa=_seq(100, "D")),  # rejected
    ]
    df, stats = dl.collect_filtered_stream(rows, target_sequences=100, max_scanned=100)
    assert list(df["sequence"]) == [
        dl.clean_vh_aa(g1),
        dl.clean_vh_aa(g2),
    ]
    assert stats["scanned"] == 4 and stats["cheap_kept"] == 2
    assert (df["source"] == "RUN1").all() and (df["isotype"] == "IGHG").all()


def test_collect_filtered_stream_early_stop():
    rows = [_row(sequence_alignment_aa=_distinct(i)) for i in range(10)]
    df, stats = dl.collect_filtered_stream(rows, target_sequences=3, max_scanned=100)
    assert stats["cheap_kept"] == 3
    assert stats["scanned"] == 3  # stopped as soon as the 3rd unique was kept


def test_collect_filtered_stream_max_scanned_cap():
    rows = [_row(meta_Species="mouse") for _ in range(1000)]  # never pass
    df, stats = dl.collect_filtered_stream(rows, target_sequences=100, max_scanned=10)
    assert stats["cheap_kept"] == 0 and stats["scanned"] == 10 and len(df) == 0


def test_choose_shards_spread():
    shards = [f"s{i:04d}.parquet" for i in range(100)]
    chosen = dl._choose_shards(shards, 10)
    assert chosen == [f"s{i:04d}.parquet" for i in range(0, 100, 10)]  # even stride
    assert dl._choose_shards(shards, 1000) == shards  # more requested than available
    assert dl._choose_shards(shards, 0) == shards  # disabled
