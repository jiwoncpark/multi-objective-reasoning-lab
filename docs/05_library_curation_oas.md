# Step 4 — Candidate library: OAS streaming + curation + loaders

**Status:** DONE (2026-06-27) — `scripts/{download_oas,build_library}.py` + `mobo_lab/data.py` + tests;
19 download tests + full suite 68/68 green; HF streaming + ANARCII validated end-to-end on real data.
**Depends on:** Step 1 (`config`)
**Unblocks:** embeddings (Step 5), oracle/initial design (Step 6), pool (Step 7), everything downstream.

## Goal

Build a curated VH candidate pool of **2^11 = 2048** sequences from OAS, streamed from the Hugging Face
Parquet mirror **`ConvergeBio/oas-unpaired`** (no manual download). The pool is the nearest-neighbor
projection library + dense "ground-truth" Pareto-front scatter; only `N_INITIAL` (~12) is evaluated to
start a run. The synthetic oracle is over the latents, so the library needs only **valid VH sequences**.

## Two-stage filter

The HF mirror carries the AIRR fields **and** the unit metadata as per-row columns (`meta_Species`,
`meta_Chain`, `meta_BSource`, `meta_BType`, `meta_Isotype`, `meta_Run`, …), so both filter levels are
row predicates:

1. **Cheap, per-row** (`row_passes`, runs on every scanned row): metadata — `meta_Species`==human,
   `meta_Chain`==heavy, `meta_BSource`∈{pbmc,leukopak}, `meta_BType`∈{unsorted/naive/memory-b-cells},
   `meta_Isotype`∈{bulk,ighm,ighd,ighg,igha}; AIRR — productive, vj_in_frame, not stop_codon; and a
   cleaned VH aa (`sequence_alignment_aa`, IMGT gaps stripped) of length 90–150 using only the 20
   standard residues.
2. **Numbering, batched** (`validate_heavy_vh`, runs on the survivors): validate each candidate with
   **ANARCII** (`anarcii`, a PyTorch antibody-numbering model — no HMMER, no license). Keep sequences
   that number as `chain_type == "H"` with `error is None` (optional `--anarcii-min-score`). This is a
   definitive numbering check, **not** a heuristic on the precomputed `ANARCI_status` string.

## Files

```text
scripts/download_oas.py             # HF stream + 2-stage filter -> data/oas_filtered.csv.gz (+ _stats.json)
scripts/build_library.py            # curate filtered set -> data/vh_library.csv (2048)
mobo_lab/data.py                    # loaders used by the package + notebooks
tests/scripts/test_download_oas.py  # offline (injected row dicts; no network, no datasets/anarcii import)
tests/scripts/test_build_library.py
tests/mobo_lab/test_data.py
```

## `scripts/download_oas.py` — collect + filter (instructor-side, network)

CLI: `python scripts/download_oas.py --config "Briney et al., 2019" --out data/oas_filtered.csv.gz`

Two collection modes, both feeding the same cheap `row_passes` filter + exact dedup, then a batched
ANARCII check:

- **Study config (recommended): `collect_across_shards`.** A study lives as many size-based Parquet
  shards (Briney 2019 = 1,525 × ~600 MB) with donors in contiguous shard ranges. We list the shards
  (`HfFileSystem`), pick `--num-shards` (default 100) **spread evenly** across them (`_choose_shards`),
  and read only the **head** of each via `pyarrow` `iter_batches(columns=NEEDED_COLUMNS)` (column
  projection ⇒ tiny I/O), capping `target//num_shards` passing rows per shard. This gives **cross-donor
  diversity** cheaply — e.g. 100 shards → ~100 distinct `meta_Run` sources.
- **Global config (`--config default`): `collect_filtered_stream`** over `load_oas_stream` (lazy
  `datasets.load_dataset(streaming=True)` → `select_columns` → optional `--shuffle`). The global stream
  is **species-ordered** (starts with rabbit/mouse), so it needs `--shuffle` for diversity — but
  streaming `.shuffle()` buffers full row-groups and **OOMs** on this 2B-row mirror, so prefer a study
  config. Early-stops at `--target-sequences` or `--max-scanned`.

Both then run **`validate_heavy_vh`** (ANARCII, batched, lazy import): keep `chain_type=="H"`,
`error is None` (`--no-anarcii` skips; `--anarcii-mode {speed,accuracy}`). Writes
`data/oas_filtered.csv.gz` (`sequence, source, isotype`) + `data/oas_filtered_stats.json`
(scanned / cheap_kept / shards_used / distinct_sources / anarcii_kept / final).

### How to obtain the OAS data

Just run the script — it reads from HF (`~/.cache/huggingface`), no manual download. Use a human study
config (`--config "Briney et al., 2019"`) so collection spans donors via the shard spread; the global
`default` config is a fallback (species-ordered, needs the memory-heavy `--shuffle`).

## `scripts/build_library.py` — curation (offline, deterministic, unchanged)

Reads `data/oas_filtered.csv.gz`; numpy k-means (`composition_features`/`kmeans`/
`keep_after_dropping_small`/`balanced_sample`/`curate`) → exact dedup → cluster → **drop singleton/rare
clusters** → balance → sample to `config.LIBRARY_SIZE` (2048) → `data/vh_library.csv`
(`sequence_id, sequence, length, source, cluster_id`). Deterministic given `--seed`.

## `mobo_lab/data.py` — loaders (unchanged)

`load_library / load_sequences / load_latents / load_initial_ids / load_true_objectives`, validating
schema, unique ids, standard residues, latent shape/range. Canonical `STANDARD_AMINO_ACIDS` lives here.

## Dependencies

`datasets` (HF streaming) and `anarcii` (numbering) added to `pyproject.toml` `[dependency-groups] dev`
— instructor-side; both are **lazy-imported** so offline tests and student notebooks don't load them.
`anarcii` is PyTorch-based and reuses the CPU torch already installed.

## Tests (offline; no network, no datasets/anarcii import)

- `test_download_oas.py`: `to_bool`, `clean_vh_aa`, `_only_standard`; `row_passes` accepts a good row
  dict and rejects one per reason (wrong `meta_*`, non-productive, out-of-frame, stop_codon, empty/short/
  long aa, ambiguous residue); `_is_valid_heavy` on synthetic ANARCII result dicts (H+no-error pass,
  light/errored/low-score reject); `collect_filtered_stream` dedup + early-stop + `max_scanned` on
  injected row dicts.
- `test_build_library.py`, `test_data.py`: unchanged from the curation/loader design.

## Verified live (not in CI)

`validate_heavy_vh([real_VH, junk, real_VH]) -> [True, False, True]`; streaming `"Briney et al., 2019"`
yields human LeukoPak rows that pass the cheap filter (25/25) and ANARCII (25/25). Metadata filtering
correctly rejects non-human (the global stream starts with rabbit shards).

## Acceptance criteria

- `uv run pytest tests/scripts/test_download_oas.py tests/scripts/test_build_library.py
  tests/mobo_lab/test_data.py` green (offline).
- Instructor run: `uv run python scripts/download_oas.py` streams from HF, prints the
  scanned→cheap→ANARCII funnel, writes `data/oas_filtered.csv.gz`; then `uv run python
  scripts/build_library.py` → `data/vh_library.csv` (2048 rows), reproducibly.
