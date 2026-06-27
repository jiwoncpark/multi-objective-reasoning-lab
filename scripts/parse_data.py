"""Parse the raw antibody VH spreadsheet into a clean, analysis-ready CSV.

The raw file ``data/vh_data.xlsx`` is a small library of antibody heavy-chain
variable-domain (VH) sequences together with four *simulated* biophysical
measurements. We do **not** feed these measurements to the lab's oracle
directly -- the oracle is a synthetic function defined later -- but we keep them
around because they tell us what realistic objective trade-offs look like when
we design that oracle.

This script does three things:

1. Reads the spreadsheet.
2. Cleans it up: snake_case column names, a numeric germline-identity column
   (the raw file stores it as a string like ``"77.1%"``), and a derived
   ``length`` column.
3. Writes the result to ``data/vh_data.csv`` and prints a short summary.

It is intentionally dependency-light (pandas + openpyxl) so it runs the same way
on a student CPU laptop as on the instructor machine.

Usage
-----
    python scripts/parse_data.py
    python scripts/parse_data.py --input data/vh_data.xlsx --output data/vh_data.csv

The parsing logic lives in :func:`parse_vh_data`, which returns a DataFrame so it
can be unit-tested without touching the filesystem (see
``tests/scripts/test_parse_data.py``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Project root is one level above this ``scripts/`` directory. We resolve paths
# relative to it so the script works regardless of the current directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "vh_data.xlsx"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "vh_data.csv"

# Map the raw spreadsheet headers to clean snake_case names. Keeping this as an
# explicit dict (rather than auto-slugifying) makes the column contract obvious
# and lets the tests assert against it. The original units are preserved in the
# new names so nothing is lost.
COLUMN_RENAMES: dict[str, str] = {
    "Sample ID": "sample_id",
    "Seq": "sequence",
    "Germline": "germline",
    "Germline_Identity%": "germline_identity_pct",
    "Tm (oC)": "tm_celsius",
    "BV Elisa score": "bv_elisa_score",
    "Affinity, Kd (nM)": "affinity_kd_nm",
    "Yield (mg per 10 mL culture)": "yield_mg_per_10ml",
}

# The 20 standard amino-acid one-letter codes. Used to validate that every
# parsed sequence is a clean protein string (no gaps, stops, or stray symbols).
STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")

# Final column order of the cleaned table. ``length`` is derived and inserted
# right after the sequence so the identity/metadata columns stay grouped.
OUTPUT_COLUMNS: list[str] = [
    "sample_id",
    "sequence",
    "length",
    "germline",
    "germline_identity_pct",
    "tm_celsius",
    "bv_elisa_score",
    "affinity_kd_nm",
    "yield_mg_per_10ml",
]


def _parse_percent(series: pd.Series) -> pd.Series:
    """Convert a percentage column to a float in *percent units* (e.g. 77.1).

    The raw file stores germline identity as strings like ``"77.1%"``. We strip
    the trailing ``%`` and cast to float. Values that are already numeric pass
    through unchanged, which keeps the function robust if the source format ever
    changes.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    cleaned = series.astype(str).str.strip().str.rstrip("%")
    return pd.to_numeric(cleaned, errors="raise")


def parse_vh_data(input_path: str | Path = DEFAULT_INPUT) -> pd.DataFrame:
    """Read the raw VH spreadsheet and return a cleaned DataFrame.

    Parameters
    ----------
    input_path:
        Path to the ``.xlsx`` file.

    Returns
    -------
    pandas.DataFrame
        One row per antibody, with the columns listed in :data:`OUTPUT_COLUMNS`.

    Raises
    ------
    FileNotFoundError
        If ``input_path`` does not exist.
    KeyError
        If the spreadsheet is missing an expected column.
    ValueError
        If sequence IDs are not unique or a sequence contains non-standard
        amino-acid characters -- both signal that the source data is not what
        the downstream lab code expects.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input spreadsheet not found: {input_path}")

    raw = pd.read_excel(input_path, sheet_name=0)

    missing = [col for col in COLUMN_RENAMES if col not in raw.columns]
    if missing:
        raise KeyError(
            f"Spreadsheet is missing expected column(s): {missing}. "
            f"Found columns: {list(raw.columns)}"
        )

    df = raw.rename(columns=COLUMN_RENAMES)[list(COLUMN_RENAMES.values())].copy()

    # Normalise text columns: strip stray whitespace and upper-case sequences.
    df["sample_id"] = df["sample_id"].astype(str).str.strip()
    df["sequence"] = df["sequence"].astype(str).str.strip().str.upper()
    df["germline"] = df["germline"].astype(str).str.strip()

    # Numeric germline identity (drop the "%").
    df["germline_identity_pct"] = _parse_percent(df["germline_identity_pct"])

    # Derived sequence length, placed next to the sequence.
    df["length"] = df["sequence"].str.len()

    # --- Validation: fail loudly if the data is not lab-ready ---------------
    if not df["sample_id"].is_unique:
        dupes = df.loc[df["sample_id"].duplicated(), "sample_id"].tolist()
        raise ValueError(f"Duplicate sample IDs found: {dupes}")

    bad_chars = (
        df["sequence"]
        .apply(lambda s: sorted(set(s) - STANDARD_AMINO_ACIDS))
        .loc[lambda x: x.str.len() > 0]
    )
    if not bad_chars.empty:
        examples = bad_chars.head(3).to_dict()
        raise ValueError(
            "Some sequences contain non-standard amino-acid characters "
            f"(row index -> offending chars): {examples}"
        )

    return df[OUTPUT_COLUMNS].reset_index(drop=True)


def summarize(df: pd.DataFrame) -> str:
    """Return a short human-readable summary of the parsed table."""
    numeric_cols = [
        "length",
        "germline_identity_pct",
        "tm_celsius",
        "bv_elisa_score",
        "affinity_kd_nm",
        "yield_mg_per_10ml",
    ]
    lines = [
        f"Parsed {len(df)} antibody VH sequences.",
        f"Unique germlines: {df['germline'].nunique()}",
        f"Sequence length: min={df['length'].min()}, "
        f"max={df['length'].max()}, median={int(df['length'].median())}",
        "",
        "Property ranges (min / median / max):",
    ]
    for col in numeric_cols:
        s = df[col]
        lines.append(
            f"  {col:24s} {s.min():10.3g} / {s.median():10.3g} / {s.max():10.3g}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to the raw .xlsx file (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Where to write the cleaned .csv (default: {DEFAULT_OUTPUT}).",
    )
    args = parser.parse_args()

    df = parse_vh_data(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(summarize(df))
    print(f"\nWrote cleaned table to: {args.output}")


if __name__ == "__main__":
    main()
