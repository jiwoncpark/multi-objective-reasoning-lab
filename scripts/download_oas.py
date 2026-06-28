"""Stream and filter OAS heavy-chain VH sequences from the Hugging Face mirror.

This instructor-only script streams the Parquet mirror ``ConvergeBio/oas-unpaired``
(OAS-unpaired, ~2.07B heavy rows) with ``datasets.load_dataset(..., streaming=True)``
and keeps only clean human heavy VH sequences. The result, a compact
``data/oas_filtered.csv.gz``, is what ``scripts/build_library.py`` curates into the
candidate library.

Why streaming: the mirror is columnar Parquet, so we ``select_columns`` the handful
of fields we need and **stop early** once enough passing sequences are collected -- we
never download the whole dataset. A streaming ``shuffle`` mixes studies so the sample
is diverse (the global stream is ordered by study/species).

Filtering is two-stage:

* **cheap, per-row** -- metadata (human heavy, desired BSource/BType/Isotype) plus AIRR
  flags (productive, in-frame, no stop codon) and a cleaned VH amino-acid sequence of
  sane length using only the 20 standard residues. These run on every scanned row.
* **numbering, batched** -- the surviving candidates are validated with **ANARCII**
  (a PyTorch antibody-numbering model; no HMMER), keeping only sequences that number
  as a heavy chain with no error. This is a definitive check, not a heuristic on the
  precomputed ``ANARCI_status`` string.

Usage
-----
    python scripts/download_oas.py --target-sequences 10240 --out data/oas_filtered.csv.gz

Only this script touches the network; curation and all tests run offline.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # allow running before the package is installed
from mobo_lab import config  # noqa: E402
from mobo_lab.data import STANDARD_AMINO_ACIDS  # noqa: E402

HF_DATASET = "ConvergeBio/oas-unpaired"

# Columns we read (a small slice of the ~120-column schema) -> far less streamed I/O.
NEEDED_COLUMNS = [
    "sequence_alignment_aa",
    "productive",
    "vj_in_frame",
    "stop_codon",
    "meta_Species",
    "meta_Chain",
    "meta_BSource",
    "meta_BType",
    "meta_Isotype",
    "meta_Run",
    "meta_Author",
]

# Per-row metadata acceptance (values matched case/space-folded).
META_VALUE_FILTERS: dict[str, set[str]] = {
    "meta_Species": {"human"},
    "meta_Chain": {"heavy"},
    "meta_BSource": {"pbmc", "leukopak"},
    "meta_BType": {"unsorted-b-cells", "naive-b-cells", "memory-b-cells"},
    "meta_Isotype": {"bulk", "ighm", "ighd", "ighg", "igha"},
}

AA_COLUMN = "sequence_alignment_aa"
DEFAULT_OUT = REPO_ROOT / "data" / "oas_filtered.csv.gz"


# --------------------------------------------------------------------------- #
# Cheap per-row filters (offline-testable)
# --------------------------------------------------------------------------- #
def _norm(value) -> str:
    return str(value).strip().lower()


def to_bool(value) -> bool:
    """Parse OAS truthy fields: bool, ``T``/``F``, ``true``/``false``, ``1``/``0``."""
    if isinstance(value, bool):
        return value
    return _norm(value) in {"t", "true", "1", "1.0", "yes", "y"}


def clean_vh_aa(seq) -> str:
    """Uppercase and strip IMGT gap characters / whitespace from a VH aa field."""
    if seq is None:
        return ""
    return re.sub(r"[\.\-\s]", "", str(seq)).upper()


def _only_standard(seq: str) -> bool:
    return len(seq) > 0 and set(seq) <= STANDARD_AMINO_ACIDS


def metadata_ok(row: dict) -> tuple[bool, str]:
    """Return ``(accepted, reason)`` for a row's ``meta_*`` columns."""
    for col, allowed in META_VALUE_FILTERS.items():
        if _norm(row.get(col)) not in allowed:
            return False, f"{col}={row.get(col)!r}"
    return True, "ok"


def row_passes(row: dict, min_len: int = 90, max_len: int = 150) -> tuple[bool, str]:
    """Apply the cheap per-row filters; return ``(passed, cleaned_sequence)``.

    Numbering validity is checked separately and in batch by :func:`validate_heavy_vh`.
    """
    ok, _ = metadata_ok(row)
    if not ok:
        return False, ""
    if not to_bool(row.get("productive")):
        return False, ""
    if not to_bool(row.get("vj_in_frame")):
        return False, ""
    if to_bool(row.get("stop_codon")):
        return False, ""
    cleaned = clean_vh_aa(row.get(AA_COLUMN))
    if not (min_len <= len(cleaned) <= max_len):
        return False, ""
    if not _only_standard(cleaned):
        return False, ""
    return True, cleaned


def collect_filtered_stream(
    rows: Iterable[dict],
    *,
    target_sequences: int,
    max_scanned: int,
    min_len: int = 90,
    max_len: int = 150,
) -> tuple[pd.DataFrame, dict]:
    """Cheap-filter + dedup a row stream, stopping at ``target_sequences`` or ``max_scanned``.

    Returns ``(DataFrame[sequence, source, isotype], stats)``. ``rows`` is consumed
    lazily, so a streaming dataset only fetches what is scanned.
    """
    seen: set[str] = set()
    records: list[dict] = []
    scanned = 0
    for row in rows:
        scanned += 1
        passed, cleaned = row_passes(row, min_len, max_len)
        if passed and cleaned not in seen:
            seen.add(cleaned)
            source = row.get("meta_Run") or row.get("meta_Author") or ""
            records.append(
                {"sequence": cleaned, "source": str(source), "isotype": row.get("meta_Isotype", "")}
            )
            if len(records) >= target_sequences:
                break
        if scanned >= max_scanned:
            break

    stats = {
        "scanned": scanned,
        "cheap_kept": len(records),
        "cheap_pass_rate": (len(records) / scanned) if scanned else 0.0,
    }
    df = pd.DataFrame(records, columns=["sequence", "source", "isotype"])
    return df, stats


# --------------------------------------------------------------------------- #
# Multi-shard collection (cross-donor diversity within one study config)
# --------------------------------------------------------------------------- #
def _choose_shards(shards: list[str], num_shards: int) -> list[str]:
    """Pick up to ``num_shards`` shards spread evenly across the list (deterministic).

    A study's donors occupy contiguous shard ranges, so an even spread samples across
    donors rather than reading one donor's shards.
    """
    if num_shards <= 0 or num_shards >= len(shards):
        return list(shards)
    stride = len(shards) / num_shards
    picked = sorted({int(i * stride) for i in range(num_shards)})
    return [shards[i] for i in picked]


def collect_across_shards(
    study: str,
    split: str = "heavy",
    *,
    target_sequences: int,
    num_shards: int = 100,
    max_rows_per_shard: int = 50_000,
    min_len: int = 90,
    max_len: int = 150,
) -> tuple[pd.DataFrame, dict, list[dict]]:
    """Collect cheap-filtered sequences from a spread of a study's Parquet shards.

    Reads only the head of each chosen shard (pyarrow column-projected row batches), so
    I/O stays bounded while the sample spans multiple donors/runs. Lazy-imports
    ``pyarrow`` + ``huggingface_hub``.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    shard_dir = f"datasets/{HF_DATASET}/data/unpaired_{split}/{study}"
    shards = sorted(p for p in fs.ls(shard_dir, detail=False) if p.endswith(".parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards under {shard_dir}")
    chosen = _choose_shards(shards, num_shards)
    per_shard = max(1, target_sequences // len(chosen))

    seen: set[str] = set()
    records: list[dict] = []
    manifest: list[dict] = []
    total_scanned = 0
    for shard in chosen:
        kept_here = scanned_here = 0
        try:
            with fs.open(shard, "rb") as fh:
                parquet = pq.ParquetFile(fh)
                for batch in parquet.iter_batches(batch_size=2048, columns=NEEDED_COLUMNS):
                    for row in batch.to_pylist():
                        scanned_here += 1
                        passed, cleaned = row_passes(row, min_len, max_len)
                        if passed and cleaned not in seen:
                            seen.add(cleaned)
                            source = row.get("meta_Run") or row.get("meta_Author") or ""
                            records.append(
                                {"sequence": cleaned, "source": str(source),
                                 "isotype": row.get("meta_Isotype", "")}
                            )
                            kept_here += 1
                        if kept_here >= per_shard or scanned_here >= max_rows_per_shard:
                            break
                    if kept_here >= per_shard or scanned_here >= max_rows_per_shard:
                        break
        except Exception as exc:  # pragma: no cover - network/shard-specific
            warnings.warn(f"shard {shard.split('/')[-1]} failed: {exc}", stacklevel=2)
        total_scanned += scanned_here
        manifest.append({"shard": shard.split("/")[-1], "kept": kept_here, "scanned": scanned_here})
        if len(records) >= target_sequences:
            break

    stats = {
        "scanned": total_scanned,
        "cheap_kept": len(records),
        "shards_used": len(manifest),
        "distinct_sources": len({r["source"] for r in records}),
    }
    df = pd.DataFrame(records, columns=["sequence", "source", "isotype"])
    return df, stats, manifest


# --------------------------------------------------------------------------- #
# ANARCII numbering validation (batched; lazy import)
# --------------------------------------------------------------------------- #
def _is_valid_heavy(result: dict, min_score: float = 0.0) -> bool:
    """A clean heavy VH numbers as chain ``H`` with no error and a high-enough score."""
    return (
        result.get("chain_type") == "H"
        and result.get("error") is None
        and float(result.get("score") or 0.0) >= min_score
    )


def validate_heavy_vh(
    sequences: Iterable[str], *, mode: str = "speed", min_score: float = 0.0, batch_size: int = 64
) -> list[bool]:
    """Return a per-sequence mask of clean heavy VHs, using ANARCII (lazy import)."""
    seqs = list(sequences)
    if not seqs:
        return []
    from anarcii import Anarcii  # lazy

    model = Anarcii(seq_type="antibody", mode=mode, cpu=True, batch_size=batch_size, verbose=False)
    numbered = model.number(seqs)
    results = list(numbered.values())  # ANARCII preserves input order
    if len(results) != len(seqs):  # pragma: no cover - defensive
        raise RuntimeError(f"ANARCII returned {len(results)} results for {len(seqs)} sequences")
    return [_is_valid_heavy(r, min_score) for r in results]


# --------------------------------------------------------------------------- #
# Streaming loader (lazy datasets import -- offline code/tests don't need it)
# --------------------------------------------------------------------------- #
def load_oas_stream(
    config_name: str = "default",
    split: str = "heavy",
    *,
    shuffle_buffer: int = 100_000,
    seed: int = config.SEED,
    shuffle: bool = False,
):
    """Return a streaming, column-projected OAS dataset iterator.

    ``shuffle`` is **off by default**: streaming ``.shuffle()`` buffers full Parquet
    row-groups and OOMs on this 2B-row mirror. Target a human study ``config_name``
    (all rows already human) and read sequentially instead. Only enable ``shuffle`` for
    the species-ordered global ``default`` config, and expect high memory use.
    """
    from datasets import load_dataset  # lazy

    ds = load_dataset(HF_DATASET, config_name, split=split, streaming=True)
    try:
        ds = ds.select_columns(NEEDED_COLUMNS)
    except Exception as exc:  # pragma: no cover - depends on datasets version
        warnings.warn(f"select_columns failed ({exc}); streaming full rows", stacklevel=2)
    if shuffle and shuffle_buffer:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    return ds


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stream + filter OAS heavy VH sequences from HF.")
    parser.add_argument("--config", default="default", help="HF config (default = all; or a study name)")
    parser.add_argument("--split", default="heavy", choices=["heavy", "light"])
    parser.add_argument("--target-sequences", type=int, default=10_240, help="early-stop threshold")
    parser.add_argument("--max-scanned", type=int, default=5_000_000, help="safety cap on rows scanned")
    parser.add_argument(
        "--num-shards",
        type=int,
        default=100,
        help="study configs only: spread the sample across this many shards (cross-donor diversity)",
    )
    parser.add_argument("--max-rows-per-shard", type=int, default=50_000)
    parser.add_argument("--shuffle-buffer", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="streaming shuffle (memory-heavy; only for the species-ordered global config)",
    )
    parser.add_argument("--min-len", type=int, default=90)
    parser.add_argument("--max-len", type=int, default=150)
    parser.add_argument("--no-anarcii", dest="anarcii", action="store_false", help="skip ANARCII check")
    parser.add_argument("--anarcii-mode", default="speed", choices=["speed", "accuracy"])
    parser.add_argument("--anarcii-min-score", type=float, default=0.0)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)

    if args.config == "default":
        # global config: sequential stream (species-ordered) + cheap filter + early stop
        stream = load_oas_stream(
            args.config, args.split, shuffle_buffer=args.shuffle_buffer, seed=args.seed,
            shuffle=args.shuffle,
        )
        df, stats = collect_filtered_stream(
            stream,
            target_sequences=args.target_sequences,
            max_scanned=args.max_scanned,
            min_len=args.min_len,
            max_len=args.max_len,
        )
    else:
        # study config: spread across shards for cross-donor diversity
        df, stats, _ = collect_across_shards(
            args.config,
            args.split,
            target_sequences=args.target_sequences,
            num_shards=args.num_shards,
            max_rows_per_shard=args.max_rows_per_shard,
            min_len=args.min_len,
            max_len=args.max_len,
        )
        print(
            f"collected {len(df):,} from {stats['shards_used']} shards "
            f"across {stats['distinct_sources']} distinct sources"
        )

    if args.anarcii and len(df):
        print(f"validating {len(df):,} candidates with ANARCII ({args.anarcii_mode})...")
        mask = validate_heavy_vh(
            df["sequence"], mode=args.anarcii_mode, min_score=args.anarcii_min_score
        )
        kept = int(sum(mask))
        stats.update({"anarcii_kept": kept, "anarcii_dropped": len(df) - kept})
        df = df[pd.Series(mask, index=df.index)].reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, compression="gzip")
    stats.update({"dataset": HF_DATASET, "config": args.config, "split": args.split, "final": len(df)})
    Path(str(out_path).replace(".csv.gz", "_stats.json")).write_text(json.dumps(stats, indent=2))

    tail = f"{len(df):,} ANARCII-clean heavy VH" if args.anarcii else f"{len(df):,} (ANARCII skipped)"
    print(
        f"scanned {stats['scanned']:,} rows -> {stats['cheap_kept']:,} cheap-filtered -> {tail} "
        f"-> {out_path}"
    )


if __name__ == "__main__":
    main()
