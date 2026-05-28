#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3 — Cell-type aware MTI prediction (main model, rewrite).

Compared to the previous version, this rewrite fixes the following:

1.  Global sequence IDs.
    miRNA_id and utr_id in the parquet chunks are CHUNK-LOCAL: the same int
    refers to different miRNAs in different chunks. The only stable cross-chunk
    key is the sequence string. We therefore build:
        mirna_seq -> mirna_global_idx
        utr3_sequence -> utr_global_idx
    and translate every chunk-local id to its global counterpart before doing
    anything else.

2.  Streaming chunk scan (no pd.concat of the full master table).
    Each chunk is read independently, translated, then accumulated into
    compact numpy int arrays. Peak memory stays well under 2 GB even on the
    Ubuntu box with 31 GB RAM.

3.  Lazy UTR encoding (no 50 GB+ memmap).
    Because the global UTR dictionary only has ~12,000 unique sequences, we
    store the raw sequences once (a few tens of MB) plus pre-encoded
    miRNA one-hot codes, and compute the seed-anchored UTR window on the fly
    inside __getitem__. The cache footprint is tens of MB total.

4.  Engineering goodies:
      --max_chunks N        : only load the first N chunks (fast Mac sanity).
      --rebuild_cache       : wipe and rebuild cache/.
      --resume              : restart from latest checkpoint_epoch_*.pt.
      --holdout_cell_types  : manual holdout (overrides random selection).
      Per-epoch full checkpoints (model + optimizer + scheduler + RNG).
      Seed-anchor hit-rate report (>>> SEED ANCHOR REPORT ... <<<).
      Cross-platform path inference (uses Path(__file__).resolve().parent.parent).

Layout (created on first run):

    <V3>/cache/
        global_maps/
            mirna_seqs.json       # list[str], indexed by mirna_global_idx
            utr_seqs.pkl          # list[str], indexed by utr_global_idx
            cell_id_to_type.json
            cell_type_to_idx.json
            chunk_translation.json   # per-chunk local->global maps
        mirna_codes_global.npy    # uint8 [n_mirna_global, MIRNA_LEN]
        seed_targets.json         # list[str], 7mer DNA, per mirna_global_idx
        splits/
            train_rows.npz        # mirna_gid, utr_gid, cell_idx, label
            val_rows.npz
            test_triple_rows.npz
            test_cell_holdout_rows.npz
        split_diagnostics.json
        seed_anchor_report.json

    <V3>/runs/<run_tag>/
        train_log.txt
        history.json
        checkpoint_epoch_NNN.pt
        last_model.pt
        best_model.pt
        final_report.json
        test_*_predictions.npz

Designed for an undergraduate FYP: clear, reproducible, paper-defensible.

Run from anywhere as:
    python <V3>/script/train_main_model.py --run_tag main_v1 --max_chunks 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
V3_ROOT = SCRIPT_DIR.parent.parent     # experimental copy lives one level deeper
DATASET_DIR = V3_ROOT / "dataset"
CACHE_DIR = V3_ROOT / "cache"
RUNS_DIR = V3_ROOT / "runs"


# ---------------------------------------------------------------------------
# Constants for sequence encoding
# ---------------------------------------------------------------------------

MIRNA_LEN = 26                  # pad miRNA seqs (which are 18-24 nt) to 26
UTR_WINDOW_LEN = 2000           # seed-anchored UTR window length
SEED_MATCH_LEN = 7              # legacy: 7mer-m8 seed length
DEFAULT_SEED_RULE = "any_7mer"  # TargetScan canonical "any 7mer" (m8 OR A1)
SEED_RULES = {"7mer-m8", "7mer-A1", "6mer", "8mer", "any_7mer", "any"}

BASE_TO_CODE = np.full(256, 4, dtype=np.uint8)
for ch, code in [("A", 0), ("C", 1), ("G", 2), ("T", 3), ("U", 3),
                 ("a", 0), ("c", 1), ("g", 2), ("t", 3), ("u", 3)]:
    BASE_TO_CODE[ord(ch)] = code

RC_DNA = str.maketrans("ACGTUacgtu", "TGCAATGCAA")


def reverse_complement_rna(seq: str) -> str:
    """Reverse-complement an RNA string (U treated as T). Returns DNA letters."""
    return seq.translate(RC_DNA)[::-1]


def get_seed_target_motifs(mirna_seq: str, rule: str = DEFAULT_SEED_RULE) -> list[str]:
    """Return candidate seed-target sequences (DNA letters) for the given rule.

    TargetScan-style rule names. Returns a LIST because some rules try multiple
    motifs (e.g. any_7mer = 7mer-m8 OR 7mer-A1) and pick whichever matches first.
    """
    seg_2_7 = mirna_seq[1:7]   # miRNA positions 2..7 (1-indexed), 6 chars
    seg_2_8 = mirna_seq[1:8]   # miRNA positions 2..8, 7 chars
    if rule == "7mer-m8":
        return [reverse_complement_rna(seg_2_8)]
    if rule == "7mer-A1":
        return [reverse_complement_rna(seg_2_7) + "A"]
    if rule == "6mer":
        return [reverse_complement_rna(seg_2_7)]
    if rule == "8mer":
        return [reverse_complement_rna(seg_2_8) + "A"]
    if rule == "any_7mer":
        return [reverse_complement_rna(seg_2_8),
                reverse_complement_rna(seg_2_7) + "A"]
    if rule == "any":
        return [reverse_complement_rna(seg_2_8) + "A",
                reverse_complement_rna(seg_2_8),
                reverse_complement_rna(seg_2_7) + "A",
                reverse_complement_rna(seg_2_7)]
    raise ValueError(f"unknown seed_rule: {rule}")


# Backward-compatible single-target API (kept for the legacy test code path)
def get_seed_target_motif(mirna_seq: str) -> str:
    return get_seed_target_motifs(mirna_seq, "7mer-m8")[0]


def find_seed_anchor(utr_seq: str, seed_target) -> int:
    """Find 5'-most exact match of any seed target in utr_seq.

    seed_target may be a single string (legacy) or a list of candidate strings
    (multi-motif rules). Returns 0-indexed position of the FIRST motif's
    earliest match, or -1 if none of the motifs match.
    """
    if seed_target is None:
        return -1
    targets = [seed_target] if isinstance(seed_target, str) else list(seed_target)
    if not targets:
        return -1
    utr_norm = utr_seq.upper().replace("U", "T")
    best = -1
    for t in targets:
        if not t:
            continue
        t_norm = t.upper().replace("U", "T")
        pos = utr_norm.find(t_norm)
        if pos < 0:
            continue
        if best < 0 or pos < best:
            best = pos
    return best


def crop_utr_seed_anchored(utr_seq: str, seed_target,
                           window_len: int) -> tuple[str, int, int]:
    """Crop a window_len window from utr_seq, centred on the 5'-most seed match.

    seed_target may be str (legacy) or list[str] (multi-motif rules).

    Returns (cropped_sequence, anchor_position_in_window, anchor_position_in_utr).
    If no motif matches, anchor_position_in_utr is -1 and we fall back to the
    5'-window (positions 0..window_len).
    """
    anchor = find_seed_anchor(utr_seq, seed_target)
    L = len(utr_seq)
    if anchor < 0:
        return utr_seq[:window_len], -1, -1
    half = window_len // 2
    start = max(0, anchor - half)
    end = start + window_len
    if end > L:
        end = L
        start = max(0, end - window_len)
    return utr_seq[start:end], anchor - start, anchor


def encode_one_hot(seq: str, length: int) -> np.ndarray:
    """Encode a sequence to a [length] uint8 code array (0..3 = ACGT, 4 = pad/N)."""
    arr = np.full(length, 4, dtype=np.uint8)
    used = seq[:length]
    if used:
        view = np.frombuffer(used.encode("ascii", errors="replace"), dtype=np.uint8)
        codes = BASE_TO_CODE[view]
        arr[: len(codes)] = codes
    return arr


def codes_to_one_hot(codes: np.ndarray) -> np.ndarray:
    out = np.zeros((4, int(codes.shape[0])), dtype=np.float32)
    valid = codes < 4
    if valid.any():
        idx = np.nonzero(valid)[0]
        out[codes[valid], idx] = 1.0
    return out


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class Logger:
    def __init__(self, path: Path | None):
        self.fh = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.fh = path.open("a", encoding="utf-8")

    def log(self, msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        if self.fh is not None:
            self.fh.write(line + "\n")
            self.fh.flush()

    def close(self) -> None:
        if self.fh is not None:
            self.fh.close()


# ---------------------------------------------------------------------------
# Chunk discovery
# ---------------------------------------------------------------------------

def discover_chunks(dataset_dir: Path, max_chunks: int | None = None) -> list[str]:
    """Return sorted chunk ids with all 4 parquet files present; optionally truncated."""
    ids = set()
    for p in dataset_dir.glob("chunk_*_samples.parquet"):
        ids.add(p.name.replace("_samples.parquet", ""))
    valid = []
    for cid in sorted(ids):
        ok = all((dataset_dir / f"{cid}_{suffix}.parquet").exists()
                 for suffix in ["samples", "miRNA_dict", "utr_dict", "cell_dict"])
        if ok:
            valid.append(cid)
    if max_chunks is not None and max_chunks > 0:
        valid = valid[:max_chunks]
    return valid


# ---------------------------------------------------------------------------
# Phase A: build global ID maps (streaming, low memory)
# ---------------------------------------------------------------------------

def build_global_maps(
    dataset_dir: Path,
    chunks: list[str],
    logger: Logger,
) -> dict[str, Any]:
    """Stream every chunk's miRNA / UTR / cell dictionary and build global maps.

    Returns dict with:
        mirna_seqs:           list[str]                   (indexed by mirna_global_idx)
        utr_seqs:             list[str]                   (indexed by utr_global_idx)
        cell_id_to_type:      dict[str, str]
        cell_type_to_idx:     dict[str, int]
        chunk_translation:    dict[chunk_id, dict] with keys
                              'mirna': dict[local_mid -> global_mid]
                              'utr':   dict[local_uid -> global_uid]

    Memory bound: only one chunk's dict is held in pandas memory at a time.
    """
    import pandas as pd

    mirna_seq_to_gid: dict[str, int] = {}
    utr_seq_to_gid: dict[str, int] = {}
    mirna_seqs: list[str] = []
    utr_seqs: list[str] = []
    cell_id_to_type: dict[str, str] = {}
    chunk_translation: dict[str, dict[str, dict[int, int]]] = {}

    for cid in chunks:
        m = pd.read_parquet(dataset_dir / f"{cid}_miRNA_dict.parquet",
                            columns=["miRNA_id", "miRNA_sequence"])
        local_m: dict[int, int] = {}
        for lid, seq in zip(m["miRNA_id"].to_numpy(), m["miRNA_sequence"].astype(str).to_numpy()):
            lid_i = int(lid)
            gid = mirna_seq_to_gid.get(seq)
            if gid is None:
                gid = len(mirna_seqs)
                mirna_seq_to_gid[seq] = gid
                mirna_seqs.append(seq)
            local_m[lid_i] = gid
        del m

        u = pd.read_parquet(dataset_dir / f"{cid}_utr_dict.parquet",
                            columns=["utr_id", "utr3_sequence"])
        local_u: dict[int, int] = {}
        for lid, seq in zip(u["utr_id"].to_numpy(), u["utr3_sequence"].astype(str).to_numpy()):
            lid_i = int(lid)
            gid = utr_seq_to_gid.get(seq)
            if gid is None:
                gid = len(utr_seqs)
                utr_seq_to_gid[seq] = gid
                utr_seqs.append(seq)
            local_u[lid_i] = gid
        del u

        c = pd.read_parquet(dataset_dir / f"{cid}_cell_dict.parquet",
                            columns=["cell_id", "cell_type"])
        for cell_id, ctype in zip(c["cell_id"].astype(str).to_numpy(),
                                  c["cell_type"].astype(str).to_numpy()):
            if cell_id in cell_id_to_type and cell_id_to_type[cell_id] != ctype:
                logger.log(f"[warn] cell_id {cell_id} cell_type mismatch "
                           f"({cell_id_to_type[cell_id]} vs {ctype}); keeping first")
            else:
                cell_id_to_type[cell_id] = ctype
        del c

        chunk_translation[cid] = {"mirna": local_m, "utr": local_u}
        logger.log(f"[global_map] {cid}: cum unique miRNAs={len(mirna_seqs)} "
                   f"UTRs={len(utr_seqs)} cells={len(cell_id_to_type)}")

    all_cell_types = sorted(set(cell_id_to_type.values()))
    cell_type_to_idx = {ct: i for i, ct in enumerate(all_cell_types)}
    # Also assign a stable integer to every unique cell_id (768 individual cells)
    # for the hierarchical cell encoding. Sorted by string for determinism.
    all_cell_ids = sorted(cell_id_to_type.keys())
    cell_id_to_idx = {cid: i for i, cid in enumerate(all_cell_ids)}
    logger.log(f"[global_map] TOTAL unique miRNAs={len(mirna_seqs)} "
               f"UTRs={len(utr_seqs)} cells={len(cell_id_to_type)} "
               f"cell_types={len(cell_type_to_idx)}")
    logger.log(f"[global_map] cell_types: {all_cell_types}")

    return {
        "mirna_seqs": mirna_seqs,
        "utr_seqs": utr_seqs,
        "cell_id_to_type": cell_id_to_type,
        "cell_type_to_idx": cell_type_to_idx,
        "cell_id_to_idx": cell_id_to_idx,
        "chunk_translation": chunk_translation,
    }


# ---------------------------------------------------------------------------
# Phase B: stream samples, translate to global ids, concat into compact arrays
# ---------------------------------------------------------------------------

def build_master_samples(
    dataset_dir: Path,
    chunks: list[str],
    gmaps: dict[str, Any],
    logger: Logger,
) -> dict[str, np.ndarray]:
    """Stream each chunk's samples table, translate to globals, return numpy arrays.

    Returns dict with:
        mirna_gid:  int32 [N]
        utr_gid:    int32 [N]
        cell_id:    object array of strings, length N
        label:      uint8 [N]
        cell_type:  object array of strings, length N  (denormalised for split)
    """
    import pandas as pd

    cell_id_to_type = gmaps["cell_id_to_type"]
    chunk_translation = gmaps["chunk_translation"]

    mirna_parts: list[np.ndarray] = []
    utr_parts: list[np.ndarray] = []
    cell_id_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    cell_type_parts: list[np.ndarray] = []

    total = 0
    for cid in chunks:
        s = pd.read_parquet(dataset_dir / f"{cid}_samples.parquet",
                            columns=["miRNA_id", "utr_id", "cell_id", "label"])
        local_m = chunk_translation[cid]["mirna"]
        local_u = chunk_translation[cid]["utr"]

        local_m_arr = np.fromiter(
            (local_m[int(x)] for x in s["miRNA_id"].to_numpy()),
            dtype=np.int32, count=len(s))
        local_u_arr = np.fromiter(
            (local_u[int(x)] for x in s["utr_id"].to_numpy()),
            dtype=np.int32, count=len(s))
        cell_id_arr = s["cell_id"].astype(str).to_numpy()
        label_arr = s["label"].astype(np.uint8).to_numpy()
        cell_type_arr = np.array(
            [cell_id_to_type.get(c, "") for c in cell_id_arr], dtype=object)

        mirna_parts.append(local_m_arr)
        utr_parts.append(local_u_arr)
        cell_id_parts.append(cell_id_arr.astype(object))
        label_parts.append(label_arr)
        cell_type_parts.append(cell_type_arr)

        total += len(s)
        logger.log(f"[master] {cid}: +{len(s):,} rows (running total {total:,})")
        del s

    out = {
        "mirna_gid": np.concatenate(mirna_parts),
        "utr_gid": np.concatenate(utr_parts),
        "cell_id": np.concatenate(cell_id_parts),
        "label": np.concatenate(label_parts),
        "cell_type": np.concatenate(cell_type_parts),
    }
    logger.log(f"[master] total rows={len(out['label']):,} "
               f"positive rate={float(out['label'].mean()):.4f}")
    return out


# ---------------------------------------------------------------------------
# Phase C: paper-style split
# ---------------------------------------------------------------------------

def _stable_seed(*parts) -> int:
    """Deterministic 32-bit seed from arbitrary parts (md5-based, not Python hash)."""
    h = hashlib.md5("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(h[:4], "little")


@dataclass
class SplitConfig:
    holdout_cell_types_n: int = 3
    holdout_cell_types: list[str] | None = None
    train_frac: float = 0.70
    val_frac: float = 0.10
    test_frac: float = 0.20
    seed: int = 42


def make_paper_split(
    master: dict[str, np.ndarray],
    cfg: SplitConfig,
    cell_type_to_idx: dict[str, int],
    logger: Logger,
    cell_id_to_idx: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Apply the paper-spec split (pair-level holdout, no leakage) and emit
    FIVE evaluation sets to support a defensible cell-aware claim:

      train               main-pool, pair-bucket=train       (used for fitting)
      val                 main-pool, pair-bucket=val         (early stopping)
      test_triple         main-pool, pair-bucket=test        (pair-level generalization)
      test_cell_holdout   holdout-cells, ALL pairs           (cell-level generalization, with pair leakage)
      test_pair_x_cell_holdout  holdout-cells, pair-bucket=test (TRUE zero-shot)

    Step 1: hold out cfg.holdout_cell_types_n cell_types completely (or use
            cfg.holdout_cell_types if given) -> the cell-holdout pool.
    Step 2: independently within (a) main pool and (b) cell-holdout pool,
            keep only (mirna_gid, cell_id) groups with BOTH labels present.
            This enforces the non-trivial-context property uniformly.
    Step 3: GLOBALLY partition all unique (mirna_gid, utr_gid) pairs (across
            both pools) into train (70%) / val (10%) / test (20%) using a
            stable md5-derived RNG. Same pair => same bucket, everywhere.
    Step 4: emit the 5 sets by intersecting (cell-pool membership) with
            (pair-bucket).

    Returns dict with:
        train, val, test_triple, test_cell_holdout, test_pair_x_cell_holdout:
            each a dict of arrays (mirna_gid, utr_gid, cell_idx, label, ...)
        diagnostics: dict
    """
    cell_id_arr = master["cell_id"]
    cell_type_arr = master["cell_type"]
    mirna_gid = master["mirna_gid"]
    utr_gid = master["utr_gid"]
    label = master["label"]

    n_orig = len(label)
    # drop rows with empty cell_type (shouldn't happen, defensive)
    keep_known = cell_type_arr != ""
    if (~keep_known).any():
        n_drop = int((~keep_known).sum())
        logger.log(f"[split] dropping {n_drop:,} rows with unknown cell_type")
        cell_id_arr = cell_id_arr[keep_known]
        cell_type_arr = cell_type_arr[keep_known]
        mirna_gid = mirna_gid[keep_known]
        utr_gid = utr_gid[keep_known]
        label = label[keep_known]

    rng = np.random.default_rng(cfg.seed)
    all_cell_types = sorted(set(cell_type_arr.tolist()))
    logger.log(f"[split] available cell_types ({len(all_cell_types)}): {all_cell_types}")

    if cfg.holdout_cell_types is not None:
        requested = set(cfg.holdout_cell_types)
        unknown = requested - set(all_cell_types)
        if unknown:
            raise ValueError(f"unknown cell_types in --holdout_cell_types: {sorted(unknown)}")
        holdout_cts = sorted(requested)
        logger.log(f"[split] MANUAL holdout cell_types ({len(holdout_cts)}): {holdout_cts}")
    else:
        n_hold = min(cfg.holdout_cell_types_n, max(0, len(all_cell_types) - 1))
        perm = rng.permutation(len(all_cell_types))
        holdout_cts = sorted([all_cell_types[i] for i in perm[:n_hold]])
        logger.log(f"[split] RANDOM holdout cell_types ({len(holdout_cts)}): {holdout_cts}")

    holdout_set = set(holdout_cts)
    main_mask_global = np.array([ct not in holdout_set for ct in cell_type_arr])
    cellhold_mask_global = ~main_mask_global
    logger.log(f"[split] cell-holdout rows={int(cellhold_mask_global.sum()):,} "
               f"main rows={int(main_mask_global.sum()):,}")

    # ---- Step 2: both-label filter, applied INDEPENDENTLY to each pool ----
    def _both_label_keep(rows_mask: np.ndarray) -> tuple[np.ndarray, int, int]:
        """Returns (keep_mask, n_groups_total, n_groups_kept) for the subset."""
        sub_m = mirna_gid[rows_mask].tolist()
        sub_c = cell_id_arr[rows_mask].tolist()
        sub_y = label[rows_mask].tolist()
        group_pn: dict[tuple[int, str], list[int]] = {}
        for m, c, y in zip(sub_m, sub_c, sub_y):
            k = (int(m), str(c))
            if k not in group_pn:
                group_pn[k] = [0, 0]
            group_pn[k][int(y)] += 1
        kept = {k for k, (n0, n1) in group_pn.items() if n0 > 0 and n1 > 0}
        local_keep = np.array([(int(m), str(c)) in kept for m, c in zip(sub_m, sub_c)])
        # lift back to global indexing
        full_keep = np.zeros(len(rows_mask), dtype=bool)
        sub_indices = np.nonzero(rows_mask)[0]
        full_keep[sub_indices[local_keep]] = True
        return full_keep, len(group_pn), len(kept)

    main_keep_mask, n_groups_main_total, n_groups_main_kept = _both_label_keep(main_mask_global)
    cellhold_keep_mask, n_groups_ch_total, n_groups_ch_kept = _both_label_keep(cellhold_mask_global)
    logger.log(f"[split] main pool (mirna, cell_id) groups: total={n_groups_main_total:,} "
               f"both-label={n_groups_main_kept:,} ({n_groups_main_kept/max(n_groups_main_total,1):.2%})")
    logger.log(f"[split] cell-holdout pool (mirna, cell_id) groups: total={n_groups_ch_total:,} "
               f"both-label={n_groups_ch_kept:,} ({n_groups_ch_kept/max(n_groups_ch_total,1):.2%})")
    logger.log(f"[split] main rows after both-label filter: {int(main_keep_mask.sum()):,}")
    logger.log(f"[split] cell-holdout rows after both-label filter: {int(cellhold_keep_mask.sum()):,}")

    # ---- Step 3: GLOBAL (mirna, utr) pair-level split using stable per-pair seed ----
    # Collect every unique pair that survives the both-label filter in EITHER pool
    # (union) so that the partition is shared across main and cell-holdout. Each
    # pair gets a deterministic md5-derived random number in [0, 1) and is bucketed
    # into train (< train_frac) / val (< train+val) / test (else).
    train_thr = cfg.train_frac
    val_thr = cfg.train_frac + cfg.val_frac

    kept_any_mask = main_keep_mask | cellhold_keep_mask
    unique_pairs = set(zip(mirna_gid[kept_any_mask].tolist(), utr_gid[kept_any_mask].tolist()))
    n_unique_pairs = len(unique_pairs)
    pair_to_bucket: dict[tuple[int, int], int] = {}
    for (mid, uid) in unique_pairs:
        local_rng = np.random.default_rng(_stable_seed(cfg.seed, "pair", mid, uid))
        r = float(local_rng.random())
        if r < train_thr:
            pair_to_bucket[(mid, uid)] = 0
        elif r < val_thr:
            pair_to_bucket[(mid, uid)] = 1
        else:
            pair_to_bucket[(mid, uid)] = 2

    pair_buckets = list(pair_to_bucket.values())
    n_pair_train = int(sum(1 for b in pair_buckets if b == 0))
    n_pair_val = int(sum(1 for b in pair_buckets if b == 1))
    n_pair_test = int(sum(1 for b in pair_buckets if b == 2))
    logger.log(f"[split] unique pairs partitioned (union): train={n_pair_train:,} "
               f"val={n_pair_val:,} test={n_pair_test:,} (total {n_unique_pairs:,})")

    # ---- Step 4: assemble the five sets ----
    def _row_buckets(rows_mask: np.ndarray) -> np.ndarray:
        sub_m = mirna_gid[rows_mask].tolist()
        sub_u = utr_gid[rows_mask].tolist()
        out_local = np.fromiter(
            (pair_to_bucket.get((int(m), int(u)), -1) for m, u in zip(sub_m, sub_u)),
            dtype=np.int8, count=int(rows_mask.sum()),
        )
        return out_local

    main_bucket = _row_buckets(main_keep_mask)
    ch_bucket = _row_buckets(cellhold_keep_mask)

    def _pack(rows_mask: np.ndarray, local_select: np.ndarray) -> dict[str, np.ndarray]:
        sub_indices = np.nonzero(rows_mask)[0][local_select]
        ct = cell_type_arr[sub_indices]
        cid_str = cell_id_arr[sub_indices]
        out = {
            "mirna_gid": mirna_gid[sub_indices].astype(np.int32),
            "utr_gid": utr_gid[sub_indices].astype(np.int32),
            "cell_idx": np.array([cell_type_to_idx[c] for c in ct], dtype=np.int32),
            "cell_type": ct.astype(object),
            "cell_id": cid_str.astype(object),
            "label": label[sub_indices].astype(np.uint8),
        }
        if cell_id_to_idx is not None:
            # missing cell_id -> -1 sentinel (model will mask to 0 if no embed)
            out["cell_id_idx"] = np.array(
                [cell_id_to_idx.get(str(c), -1) for c in cid_str], dtype=np.int32)
        return out

    train_d = _pack(main_keep_mask, main_bucket == 0)
    val_d = _pack(main_keep_mask, main_bucket == 1)
    test_triple_d = _pack(main_keep_mask, main_bucket == 2)
    test_cell_holdout_d = _pack(cellhold_keep_mask, np.ones(len(ch_bucket), dtype=bool))
    test_pair_x_cell_holdout_d = _pack(cellhold_keep_mask, ch_bucket == 2)

    def _pr(d: dict[str, np.ndarray]) -> float:
        return float(d["label"].mean()) if len(d["label"]) else 0.0

    splits_meta = {
        "train": {"n": int(len(train_d["label"])), "positive_rate": _pr(train_d)},
        "val": {"n": int(len(val_d["label"])), "positive_rate": _pr(val_d)},
        "test_triple": {"n": int(len(test_triple_d["label"])), "positive_rate": _pr(test_triple_d)},
        "test_cell_holdout": {"n": int(len(test_cell_holdout_d["label"])), "positive_rate": _pr(test_cell_holdout_d)},
        "test_pair_x_cell_holdout": {"n": int(len(test_pair_x_cell_holdout_d["label"])),
                                     "positive_rate": _pr(test_pair_x_cell_holdout_d)},
    }

    diag = {
        "n_original_samples": int(n_orig),
        "holdout_cell_types": holdout_cts,
        "main_cell_types": sorted(set(all_cell_types) - holdout_set),
        "n_groups_main_total": int(n_groups_main_total),
        "n_groups_main_both_label": int(n_groups_main_kept),
        "pct_groups_main_both_label": float(n_groups_main_kept / max(n_groups_main_total, 1)),
        "n_groups_cellhold_total": int(n_groups_ch_total),
        "n_groups_cellhold_both_label": int(n_groups_ch_kept),
        "pct_groups_cellhold_both_label": float(n_groups_ch_kept / max(n_groups_ch_total, 1)),
        "split_scheme": "pair_level_holdout_v2_5sets",
        "n_unique_pairs_union": int(n_unique_pairs),
        "n_unique_pairs_per_bucket": {
            "train": n_pair_train, "val": n_pair_val, "test": n_pair_test,
        },
        "splits": splits_meta,
    }
    logger.log(
        f"[split] sizes: train={splits_meta['train']['n']:,} "
        f"val={splits_meta['val']['n']:,} "
        f"test_triple={splits_meta['test_triple']['n']:,} "
        f"test_cell_holdout={splits_meta['test_cell_holdout']['n']:,} "
        f"test_pair_x_cell_holdout={splits_meta['test_pair_x_cell_holdout']['n']:,}"
    )
    logger.log(
        f"[split] pos rates: train={splits_meta['train']['positive_rate']:.4f} "
        f"val={splits_meta['val']['positive_rate']:.4f} "
        f"test_triple={splits_meta['test_triple']['positive_rate']:.4f} "
        f"test_cell_holdout={splits_meta['test_cell_holdout']['positive_rate']:.4f} "
        f"test_pair_x_cell_holdout={splits_meta['test_pair_x_cell_holdout']['positive_rate']:.4f}"
    )

    return {
        "train": train_d,
        "val": val_d,
        "test_triple": test_triple_d,
        "test_cell_holdout": test_cell_holdout_d,
        "test_pair_x_cell_holdout": test_pair_x_cell_holdout_d,
        "diagnostics": diag,
    }


# ---------------------------------------------------------------------------
# Phase D: encode global miRNA / seed-target dictionaries
# ---------------------------------------------------------------------------

def encode_global_dictionaries(
    gmaps: dict[str, Any],
    splits: dict[str, Any],
    cache_dir: Path,
    logger: Logger,
    seed_rule: str = DEFAULT_SEED_RULE,
) -> dict[str, Any]:
    """Encode global miRNA codes + seed targets; compute seed-anchor hit rate over
    every (utr_gid, mirna_gid) pair that appears anywhere in train+val+test.

    seed_targets are stored as list[list[str]] -- one list of candidate target
    motifs per mirna_global_idx. find_seed_anchor() picks the earliest match
    among them.
    """
    mirna_seqs = gmaps["mirna_seqs"]
    utr_seqs = gmaps["utr_seqs"]
    n_mirna = len(mirna_seqs)

    mirna_codes_global = np.zeros((n_mirna, MIRNA_LEN), dtype=np.uint8)
    seed_targets: list[list[str]] = [[] for _ in range(n_mirna)]
    for i, seq in enumerate(mirna_seqs):
        mirna_codes_global[i] = encode_one_hot(seq, MIRNA_LEN)
        seed_targets[i] = get_seed_target_motifs(seq, seed_rule)
    np.save(cache_dir / "mirna_codes_global.npy", mirna_codes_global)
    (cache_dir / "seed_targets.json").write_text(
        json.dumps(seed_targets), encoding="utf-8")
    (cache_dir / "seed_rule.txt").write_text(seed_rule, encoding="utf-8")
    logger.log(f"[encode] seed_rule={seed_rule} (each mirna has "
               f"{len(seed_targets[0]) if n_mirna else 0} candidate target(s))")

    # seed-anchor hit rate report (over the union of pairs in all splits)
    seen_pairs: set[tuple[int, int]] = set()
    for name in ["train", "val", "test_triple", "test_cell_holdout", "test_pair_x_cell_holdout"]:
        sp = splits[name]
        for m, u in zip(sp["mirna_gid"].tolist(), sp["utr_gid"].tolist()):
            seen_pairs.add((m, u))

    n_pairs = len(seen_pairs)
    n_hit = 0
    n_fallback = 0
    for (m, u) in seen_pairs:
        anc = find_seed_anchor(utr_seqs[u], seed_targets[m])
        if anc >= 0:
            n_hit += 1
        else:
            n_fallback += 1
    hit_rate = n_hit / max(n_pairs, 1)

    report = {
        "seed_rule": seed_rule,
        "n_unique_pairs": n_pairs,
        "n_seed_anchor_found": n_hit,
        "n_seed_anchor_fallback": n_fallback,
        "seed_anchor_hit_rate": hit_rate,
        "seed_anchor_fallback_rate": 1.0 - hit_rate,
    }
    (cache_dir / "seed_anchor_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    logger.log(f">>> SEED ANCHOR REPORT [{seed_rule}]: pairs={n_pairs:,} hit={n_hit:,} "
               f"fallback={n_fallback:,} hit_rate={hit_rate:.4f} "
               f"fallback_rate={1.0 - hit_rate:.4f} <<<")
    return report


# ---------------------------------------------------------------------------
# Cache persistence
# ---------------------------------------------------------------------------

def write_cache(
    gmaps: dict[str, Any],
    splits: dict[str, Any],
    cache_dir: Path,
    logger: Logger,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "global_maps").mkdir(exist_ok=True)
    (cache_dir / "splits").mkdir(exist_ok=True)

    (cache_dir / "global_maps" / "mirna_seqs.json").write_text(
        json.dumps(gmaps["mirna_seqs"]), encoding="utf-8")
    with open(cache_dir / "global_maps" / "utr_seqs.pkl", "wb") as fh:
        pickle.dump(gmaps["utr_seqs"], fh, protocol=pickle.HIGHEST_PROTOCOL)
    (cache_dir / "global_maps" / "cell_id_to_type.json").write_text(
        json.dumps(gmaps["cell_id_to_type"]), encoding="utf-8")
    (cache_dir / "global_maps" / "cell_type_to_idx.json").write_text(
        json.dumps(gmaps["cell_type_to_idx"]), encoding="utf-8")
    if "cell_id_to_idx" in gmaps:
        (cache_dir / "global_maps" / "cell_id_to_idx.json").write_text(
            json.dumps(gmaps["cell_id_to_idx"]), encoding="utf-8")
    # chunk_translation keys are ints in dicts → stringify for JSON
    ct_serial = {
        cid: {
            "mirna": {str(k): v for k, v in tr["mirna"].items()},
            "utr": {str(k): v for k, v in tr["utr"].items()},
        }
        for cid, tr in gmaps["chunk_translation"].items()
    }
    (cache_dir / "global_maps" / "chunk_translation.json").write_text(
        json.dumps(ct_serial), encoding="utf-8")

    for name in ["train", "val", "test_triple", "test_cell_holdout", "test_pair_x_cell_holdout"]:
        d = splits[name]
        extra = {}
        if "cell_id_idx" in d:
            extra["cell_id_idx"] = d["cell_id_idx"]
        np.savez(cache_dir / "splits" / f"{name}_rows.npz",
                 mirna_gid=d["mirna_gid"],
                 utr_gid=d["utr_gid"],
                 cell_idx=d["cell_idx"],
                 label=d["label"],
                 cell_type=d["cell_type"],
                 cell_id=d["cell_id"],
                 **extra)
    (cache_dir / "split_diagnostics.json").write_text(
        json.dumps(splits["diagnostics"], indent=2), encoding="utf-8")
    logger.log(f"[cache] wrote cache to {cache_dir}")


def cache_exists(cache_dir: Path, expected_seed_rule: str | None = None) -> bool:
    """Return True only if the cache is complete AND (if expected_seed_rule is given)
    matches that rule. If a mismatch is found, return False so the caller knows
    to rebuild."""
    if not (cache_dir / "global_maps" / "mirna_seqs.json").exists():
        return False
    if not (cache_dir / "global_maps" / "utr_seqs.pkl").exists():
        return False
    if not (cache_dir / "mirna_codes_global.npy").exists():
        return False
    if not (cache_dir / "seed_targets.json").exists():
        return False
    for name in ["train", "val"]:
        if not (cache_dir / "splits" / f"{name}_rows.npz").exists():
            return False
    if expected_seed_rule is not None:
        rule_file = cache_dir / "seed_rule.txt"
        if not rule_file.exists():
            return False  # legacy cache without seed_rule.txt; force rebuild
        if rule_file.read_text(encoding="utf-8").strip() != expected_seed_rule:
            return False
    return True


def load_cache(cache_dir: Path, logger: Logger) -> dict[str, Any]:
    mirna_seqs = json.loads(
        (cache_dir / "global_maps" / "mirna_seqs.json").read_text(encoding="utf-8"))
    with open(cache_dir / "global_maps" / "utr_seqs.pkl", "rb") as fh:
        utr_seqs = pickle.load(fh)
    cell_type_to_idx = json.loads(
        (cache_dir / "global_maps" / "cell_type_to_idx.json").read_text(encoding="utf-8"))
    cell_id_to_idx_path = cache_dir / "global_maps" / "cell_id_to_idx.json"
    cell_id_to_idx = json.loads(cell_id_to_idx_path.read_text(encoding="utf-8")) \
        if cell_id_to_idx_path.exists() else None
    mirna_codes_global = np.load(cache_dir / "mirna_codes_global.npy")
    seed_targets = json.loads(
        (cache_dir / "seed_targets.json").read_text(encoding="utf-8"))
    n_cell_ids = len(cell_id_to_idx) if cell_id_to_idx is not None else 0
    logger.log(f"[cache] loaded: n_mirna={len(mirna_seqs)} n_utr={len(utr_seqs)} "
               f"n_cell_types={len(cell_type_to_idx)} n_cell_ids={n_cell_ids}")
    splits: dict[str, dict[str, np.ndarray]] = {}
    for name in ["train", "val", "test_triple", "test_cell_holdout", "test_pair_x_cell_holdout"]:
        p = cache_dir / "splits" / f"{name}_rows.npz"
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=True)
        splits[name] = {k: z[k] for k in z.files}
        logger.log(f"[cache] split {name}: n={len(splits[name]['label']):,} "
                   f"pos_rate={float(splits[name]['label'].mean()):.4f}")
    return {
        "mirna_seqs": mirna_seqs,
        "utr_seqs": utr_seqs,
        "cell_type_to_idx": cell_type_to_idx,
        "cell_id_to_idx": cell_id_to_idx,
        "mirna_codes_global": mirna_codes_global,
        "seed_targets": seed_targets,
        "splits": splits,
    }


# ---------------------------------------------------------------------------
# Dataset (lazy seed-anchored UTR encoding on each __getitem__)
# ---------------------------------------------------------------------------

class LazyMTIDataset:
    """Pure-Python dataset compatible with torch.utils.data.DataLoader.

    Holds:
        rows: dict[name -> np.ndarray] (mirna_gid, utr_gid, cell_idx, label)
        global_assets: mirna_codes_global (uint8), seed_targets (list[str]), utr_seqs (list[str])

    __getitem__ does:
        - look up pre-encoded mirna codes
        - look up utr sequence + seed target
        - crop seed-anchored window, encode, one-hot

    One-hot conversion is done here (uint8 -> float32 [4, L]); collate just stacks.
    """

    def __init__(self, rows: dict[str, np.ndarray],
                 mirna_codes_global: np.ndarray,
                 seed_targets: list[str],
                 utr_seqs: list[str]):
        self.mirna_gid = rows["mirna_gid"].astype(np.int64)
        self.utr_gid = rows["utr_gid"].astype(np.int64)
        self.cell_idx = rows["cell_idx"].astype(np.int64)
        # cell_id_idx may be absent in older caches; fall back to -1 sentinel
        if "cell_id_idx" in rows:
            self.cell_id_idx = rows["cell_id_idx"].astype(np.int64)
        else:
            self.cell_id_idx = np.full(len(self.cell_idx), -1, dtype=np.int64)
        self.label = rows["label"].astype(np.float32)
        self.mirna_codes_global = mirna_codes_global
        self.seed_targets = seed_targets
        self.utr_seqs = utr_seqs

    def __len__(self) -> int:
        return int(self.label.shape[0])

    def __getitem__(self, idx: int):
        mid = int(self.mirna_gid[idx])
        uid = int(self.utr_gid[idx])
        # miRNA one-hot
        mirna_codes = self.mirna_codes_global[mid]
        mirna_oh = codes_to_one_hot(mirna_codes)
        # UTR window: seed-anchored crop -> encode -> one-hot
        utr_seq = self.utr_seqs[uid]
        seed_target = self.seed_targets[mid]
        cropped, _, _ = crop_utr_seed_anchored(utr_seq, seed_target, UTR_WINDOW_LEN)
        utr_codes = encode_one_hot(cropped, UTR_WINDOW_LEN)
        utr_oh = codes_to_one_hot(utr_codes)
        v_len = min(len(cropped), UTR_WINDOW_LEN)
        return (
            mirna_oh,
            utr_oh,
            np.int32(v_len),
            np.int64(self.cell_idx[idx]),
            np.int64(self.cell_id_idx[idx]),
            np.float32(self.label[idx]),
        )


def collate_batch(batch):
    import torch
    mirna_x = torch.from_numpy(np.stack([x[0] for x in batch]))
    utr_x = torch.from_numpy(np.stack([x[1] for x in batch]))
    utr_vlen = torch.from_numpy(np.asarray([x[2] for x in batch], dtype=np.int32))
    cell_x = torch.from_numpy(np.asarray([x[3] for x in batch], dtype=np.int64))
    cell_id_x = torch.from_numpy(np.asarray([x[4] for x in batch], dtype=np.int64))
    y = torch.from_numpy(np.asarray([x[5] for x in batch], dtype=np.float32))
    return mirna_x, utr_x, utr_vlen, cell_x, cell_id_x, y


# ---------------------------------------------------------------------------
# Model: miRNA + UTR (seed-anchored, attention-pooled) + cell embedding
# ---------------------------------------------------------------------------

def make_model(n_cell_types: int, model_kind: str = "full",
               cell_embed_dim: int = 32, cell_encoding: str = "cell_type",
               n_cell_ids: int = 0):
    """model_kind controls architecture (how cell is injected):
      'full'      : miRNA seq + UTR attention-pooled + cell embedding concat at head
      'no_cell'   : miRNA seq + UTR attention-pooled                          (no cell)
      'seq_only'  : alias of no_cell
      'no_attn'   : miRNA seq + UTR MEAN-pooled + cell embedding concat at head
      'cell_only' : cell embedding only                                       (lower bound)
      'film'      : miRNA seq + UTR attention-pooled, cell injected as FiLM on
                    the final UTR feature map only (single-layer FiLM).
      'film_plus' : 'film' upgraded with two extras:
                    (C1) cell-conditioned attention bias added to the attention
                         logits (cell directly tells "look at these UTR positions"),
                    (C2) multi-layer FiLM: cell modulates the UTR feature map
                         after EACH of the three Conv stages, not just the last.
                    Tests whether deeper / wider cell injection beats single FiLM.

    cell_encoding controls HOW the cell embedding vector is computed:
      'cell_type' : nn.Embedding(16, D)  on cell_type_idx     -- shared per type only
      'hier'      : nn.Embedding(16, D)  on cell_type_idx
                  + nn.Embedding(768, D) on cell_id_idx (init=0, large weight decay)
                    -- shared type-level info + tiny per-individual offset (H1)
    """
    import torch
    from torch import nn
    import torch.nn.functional as F

    USE_SEQ = model_kind in {"full", "no_cell", "seq_only", "no_attn", "film", "film_plus"}
    USE_CELL = model_kind in {"full", "cell_only", "no_attn", "film", "film_plus"}
    USE_ATTN = model_kind in {"full", "no_cell", "seq_only", "film", "film_plus"}  # mean-pool when False
    USE_FILM = model_kind in {"film", "film_plus"}
    USE_MULTI_FILM = (model_kind == "film_plus")            # C2
    USE_CELL_ATTN_BIAS = (model_kind == "film_plus")        # C1
    CELL_CONCAT_AT_HEAD = USE_CELL and not USE_FILM  # film/film_plus inject cell ONLY via FiLM
    USE_HIER = (cell_encoding == "hier") and USE_CELL
    if USE_HIER and n_cell_ids <= 0:
        raise ValueError("cell_encoding='hier' requires n_cell_ids > 0 (rebuild cache)")

    class CellAwareMTI(nn.Module):
        def __init__(self):
            super().__init__()
            head_in = 0
            if USE_SEQ:
                self.mirna_branch = nn.Sequential(
                    nn.Conv1d(4, 32, kernel_size=7, padding=3),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.Conv1d(32, 64, kernel_size=5, padding=2),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                )
                if USE_MULTI_FILM:
                    # split UTR branch into 3 stages so we can FiLM between them (C2)
                    self.utr_stage1 = nn.Sequential(
                        nn.Conv1d(4, 32, kernel_size=9, padding=4),
                        nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),   # -> [B,32,500]
                    )
                    self.utr_stage2 = nn.Sequential(
                        nn.Conv1d(32, 64, kernel_size=7, padding=3),
                        nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),   # -> [B,64,125]
                    )
                    self.utr_stage3 = nn.Sequential(
                        nn.Conv1d(64, 128, kernel_size=5, padding=2),
                        nn.BatchNorm1d(128), nn.ReLU(),                    # -> [B,128,125]
                    )
                else:
                    self.utr_branch = nn.Sequential(
                        nn.Conv1d(4, 32, kernel_size=9, padding=4),
                        nn.BatchNorm1d(32),
                        nn.ReLU(),
                        nn.MaxPool1d(4),    # 2000 -> 500
                        nn.Conv1d(32, 64, kernel_size=7, padding=3),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        nn.MaxPool1d(4),    # 500 -> 125
                        nn.Conv1d(64, 128, kernel_size=5, padding=2),
                        nn.BatchNorm1d(128),
                        nn.ReLU(),
                    )
                self.mirna_pool = nn.AdaptiveAvgPool1d(1)
                if USE_ATTN:
                    self.mirna_to_q = nn.Linear(64, 128)
                else:
                    self.utr_pool = nn.AdaptiveAvgPool1d(1)
                head_in += 64 + 128
                self.utr_downsample = 16   # two MaxPool1d(4)
            if USE_CELL:
                self.cell_embedding = nn.Embedding(n_cell_types, cell_embed_dim)
                if USE_HIER:
                    # individual cell offset, init to 0 -> starts as pure cell_type
                    self.cell_id_embedding = nn.Embedding(n_cell_ids, cell_embed_dim)
                    nn.init.zeros_(self.cell_id_embedding.weight)
                if CELL_CONCAT_AT_HEAD:
                    head_in += cell_embed_dim
            if USE_FILM and not USE_MULTI_FILM:
                # 'film': single-layer FiLM on the final 128-channel UTR map
                self.film_gen = nn.Sequential(
                    nn.Linear(cell_embed_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 2 * 128),
                )
            if USE_MULTI_FILM:
                # 'film_plus' C2: one FiLM generator per UTR stage
                self.film_gen_s1 = nn.Linear(cell_embed_dim, 2 * 32)
                self.film_gen_s2 = nn.Linear(cell_embed_dim, 2 * 64)
                self.film_gen_s3 = nn.Linear(cell_embed_dim, 2 * 128)
                # zero-init the bias-half (β) and small-init the scale-half (γ) so each
                # FiLM starts close to identity; let gradients build it up
                for fg in [self.film_gen_s1, self.film_gen_s2, self.film_gen_s3]:
                    nn.init.zeros_(fg.bias)
                    nn.init.uniform_(fg.weight, -0.01, 0.01)
            if USE_CELL_ATTN_BIAS:
                # 'film_plus' C1: cell-derived additive bias over UTR positions in attention.
                # L_attn = UTR_WINDOW_LEN // 16 (two MaxPool1d(4) layers).
                self.cell_attn_bias = nn.Linear(cell_embed_dim, UTR_WINDOW_LEN // 16)
                nn.init.zeros_(self.cell_attn_bias.bias)
                nn.init.uniform_(self.cell_attn_bias.weight, -0.01, 0.01)
            assert head_in > 0, "no active branches"
            self.head = nn.Sequential(
                nn.Linear(head_in, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 1),
            )
            self.use_seq = USE_SEQ
            self.use_cell = USE_CELL
            self.use_attn = USE_ATTN
            self.use_film = USE_FILM
            self.use_multi_film = USE_MULTI_FILM
            self.use_cell_attn_bias = USE_CELL_ATTN_BIAS
            self.use_hier = USE_HIER
            self.cell_concat_at_head = CELL_CONCAT_AT_HEAD
            self.n_cell_ids = n_cell_ids

        def _build_utr_attn_mask(self, utr_vlen, u_seq_len: int):
            # pooled-coord valid length = ceil(valid_len / downsample)
            valid_pooled = (utr_vlen.float() / self.utr_downsample).ceil().long()
            valid_pooled = torch.clamp(valid_pooled, min=1, max=u_seq_len)
            arange = torch.arange(u_seq_len, device=utr_vlen.device).unsqueeze(0)
            return arange < valid_pooled.unsqueeze(1)

        def forward(self, mirna_x, utr_x, utr_vlen, cell_x, cell_id_x=None):
            parts = []
            # cell embedding lookup early so FiLM can use it
            cell_emb = None
            if self.use_cell:
                cell_safe = torch.clamp(cell_x, min=0)
                cell_emb = self.cell_embedding(cell_safe)                   # [B, D_c]
                if self.use_hier:
                    # H1 hierarchical: cell_emb = cell_type_emb + cell_id_offset
                    # cell_id_idx may be -1 for unseen cells (or absent); mask those to 0
                    if cell_id_x is None:
                        # caller didn't supply cell_id; act as pure cell_type encoding
                        pass
                    else:
                        cid_valid = cell_id_x >= 0                          # [B]
                        cid_safe = torch.clamp(cell_id_x, min=0)
                        id_off = self.cell_id_embedding(cid_safe)           # [B, D_c]
                        id_off = id_off * cid_valid.float().unsqueeze(-1)   # zero-out invalid
                        cell_emb = cell_emb + id_off
            if self.use_seq:
                m_seq = self.mirna_branch(mirna_x)                           # [B, 64, L_m]
                m = self.mirna_pool(m_seq).squeeze(-1)                       # [B, 64]

                # === UTR branch (with optional multi-layer FiLM) ===
                if self.use_multi_film and cell_emb is not None:
                    h = self.utr_stage1(utr_x)                               # [B, 32, 500]
                    g, b = self.film_gen_s1(cell_emb).chunk(2, dim=-1)
                    h = h * (1.0 + g.unsqueeze(-1)) + b.unsqueeze(-1)
                    h = self.utr_stage2(h)                                   # [B, 64, 125]
                    g, b = self.film_gen_s2(cell_emb).chunk(2, dim=-1)
                    h = h * (1.0 + g.unsqueeze(-1)) + b.unsqueeze(-1)
                    h = self.utr_stage3(h)                                   # [B, 128, 125]
                    g, b = self.film_gen_s3(cell_emb).chunk(2, dim=-1)
                    u_seq = h * (1.0 + g.unsqueeze(-1)) + b.unsqueeze(-1)
                else:
                    u_seq = self.utr_branch(utr_x)                           # [B, 128, L_u/16]
                    # single-layer FiLM (only when use_film and not multi)
                    if self.use_film and cell_emb is not None:
                        film_params = self.film_gen(cell_emb)                # [B, 256]
                        gamma, beta = film_params.chunk(2, dim=-1)           # each [B, 128]
                        u_seq = u_seq * (1.0 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)

                # === Pool UTR ===
                if self.use_attn:
                    q = self.mirna_to_q(m).unsqueeze(1)                      # [B, 1, 128]
                    attn = torch.bmm(q, u_seq).squeeze(1)                    # [B, L_u/16]
                    # C1: add cell-derived position bias to attention logits
                    if self.use_cell_attn_bias and cell_emb is not None:
                        attn = attn + self.cell_attn_bias(cell_emb)          # [B, L_u/16]
                    mask = self._build_utr_attn_mask(utr_vlen, u_seq.shape[-1])
                    attn = attn.masked_fill(~mask, float("-inf"))
                    all_masked = (~mask).all(dim=-1)
                    if all_masked.any():
                        attn[all_masked, 0] = 0.0
                    w = F.softmax(attn, dim=-1)
                    u = torch.bmm(u_seq, w.unsqueeze(-1)).squeeze(-1)        # [B, 128]
                else:
                    mask = self._build_utr_attn_mask(utr_vlen, u_seq.shape[-1])  # [B, L]
                    m_f = mask.float().unsqueeze(1)                          # [B, 1, L]
                    u = (u_seq * m_f).sum(dim=-1) / m_f.sum(dim=-1).clamp(min=1.0)
                parts.append(m)
                parts.append(u)
            if self.cell_concat_at_head and cell_emb is not None:
                parts.append(cell_emb)
            z = torch.cat(parts, dim=1)
            return self.head(z).squeeze(1)

    return CellAwareMTI()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def precision_recall_curve_np(y_true: np.ndarray, y_score: np.ndarray):
    order = np.argsort(-y_score)
    yt = y_true[order]
    tp = np.cumsum(yt)
    fp = np.cumsum(1 - yt)
    p_tot = float(yt.sum())
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(p_tot, 1.0)
    return precision, recall


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.sum() == 0:
        return 0.0
    p, r = precision_recall_curve_np(y_true, y_score)
    r_prev = np.concatenate([[0.0], r[:-1]])
    return float(((r - r_prev) * p).sum())


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(y_score)
    ranks = np.empty(len(y_score), dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    # tie correction
    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg_r = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg_r
        i = j + 1
    rank_sum_pos = ranks[y_true == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def best_f1(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float, float, float]:
    thr_grid = np.linspace(0.05, 0.95, 91)
    best = (0.0, 0.5, 0.0, 0.0)
    for thr in thr_grid:
        y_pred = (y_score >= thr).astype(np.int64)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        if f1 > best[0]:
            best = (f1, float(thr), p, r)
    return best


def precision_at_top_k(y_true: np.ndarray, y_score: np.ndarray, k_frac: float) -> float:
    n = len(y_true)
    k = max(1, int(round(n * k_frac)))
    idx = np.argpartition(-y_score, k - 1)[:k]
    return float(y_true[idx].sum() / k)


def evaluate(model, loader, device):
    import torch
    from torch import nn
    model.eval()
    ys, ss, cs = [], [], []
    loss_sum, n_seen = 0.0, 0
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    with torch.no_grad():
        for mirna_x, utr_x, utr_vlen, cell_x, cell_id_x, y in loader:
            mirna_x = mirna_x.to(device, non_blocking=True)
            utr_x = utr_x.to(device, non_blocking=True)
            utr_vlen = utr_vlen.to(device, non_blocking=True)
            cell_x = cell_x.to(device, non_blocking=True)
            cell_id_x = cell_id_x.to(device, non_blocking=True)
            y_dev = y.to(device, non_blocking=True)
            logits = model(mirna_x, utr_x, utr_vlen, cell_x, cell_id_x)
            loss_sum += float(bce(logits, y_dev).detach().cpu().item())
            n_seen += y.shape[0]
            ss.append(torch.sigmoid(logits).detach().cpu().numpy())
            ys.append(y.numpy())
            cs.append(cell_x.detach().cpu().numpy())
    y_true = np.concatenate(ys)
    y_score = np.concatenate(ss)
    cells = np.concatenate(cs)
    f1, thr, p_at_thr, r_at_thr = best_f1(y_true, y_score)
    return {
        "val_loss_mean": loss_sum / max(n_seen, 1),
        "pr_auc": average_precision(y_true, y_score),
        "roc_auc": roc_auc(y_true, y_score),
        "best_f1": f1,
        "best_threshold": thr,
        "precision_at_thr": p_at_thr,
        "recall_at_thr": r_at_thr,
        "p_at_top_1pct": precision_at_top_k(y_true, y_score, 0.01),
        "p_at_top_5pct": precision_at_top_k(y_true, y_score, 0.05),
        "p_at_top_10pct": precision_at_top_k(y_true, y_score, 0.10),
        "positive_rate": float(y_true.mean()),
        "n": int(len(y_true)),
        "_y_true": y_true,
        "_y_score": y_score,
        "_cell": cells,
    }


def per_cell_metrics(eval_dict, idx_to_cell_type, min_n: int = 50):
    y = eval_dict["_y_true"]
    s = eval_dict["_y_score"]
    c = eval_dict["_cell"]
    rows = []
    for cell_idx in np.unique(c):
        mask = c == cell_idx
        if mask.sum() < min_n:
            continue
        yi = y[mask]; si = s[mask]
        if yi.sum() == 0 or yi.sum() == len(yi):
            continue
        rows.append({
            "cell_idx": int(cell_idx),
            "cell_type": idx_to_cell_type.get(int(cell_idx), "?"),
            "n": int(len(yi)),
            "positive_rate": float(yi.mean()),
            "pr_auc": average_precision(yi, si),
            "roc_auc": roc_auc(yi, si),
        })
    rows.sort(key=lambda r: -r["n"])
    return rows


# ---------------------------------------------------------------------------
# Device + training loop
# ---------------------------------------------------------------------------

def resolve_device(arg: str):
    import torch
    if arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(arg)


def train_one_epoch(model, loader, loss_fn, opt, device, logger, log_every: int, epoch: int):
    model.train()
    total_loss = 0.0
    n_batches = 0
    total_batches = len(loader)
    t0 = time.time()
    for bidx, (mirna_x, utr_x, utr_vlen, cell_x, cell_id_x, y) in enumerate(loader, start=1):
        mirna_x = mirna_x.to(device, non_blocking=True)
        utr_x = utr_x.to(device, non_blocking=True)
        utr_vlen = utr_vlen.to(device, non_blocking=True)
        cell_x = cell_x.to(device, non_blocking=True)
        cell_id_x = cell_id_x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        opt.zero_grad()
        logits = model(mirna_x, utr_x, utr_vlen, cell_x, cell_id_x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        total_loss += float(loss.detach().cpu().item())
        n_batches += 1
        if log_every > 0 and (bidx % log_every == 0 or bidx == total_batches):
            elapsed = time.time() - t0
            eta = elapsed / max(bidx, 1) * max(total_batches - bidx, 0)
            logger.log(
                f"[train] epoch={epoch} batch={bidx:,}/{total_batches:,} "
                f"running_loss={total_loss / n_batches:.4f} elapsed={elapsed:.1f}s eta={eta:.1f}s"
            )
    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Config + argparse
# ---------------------------------------------------------------------------

@dataclass
class Config:
    run_tag: str
    model_kind: str = "full"
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 15
    patience: int = 3
    cell_embed_dim: int = 32
    num_workers: int = 4
    log_every_batches: int = 200
    seed: int = 42
    device: str = "auto"
    rebuild_cache: bool = False
    holdout_cell_types_n: int = 3
    holdout_cell_types: list[str] | None = None
    pos_weight_mode: str = "auto"
    resume: bool = False
    keep_all_epochs: bool = True
    max_chunks: int = 0    # 0 means "use all"
    seed_rule: str = DEFAULT_SEED_RULE
    cell_encoding: str = "cell_type"   # 'cell_type' (16 lookup) or 'hier' (H1: type + id offset)
    cell_id_weight_decay: float = 1e-2  # higher wd for cell_id_emb to prevent overfit


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--run_tag", default="main_v1")
    p.add_argument("--model", choices=["full", "no_cell", "seq_only", "no_attn", "cell_only", "film", "film_plus"], default="full")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--cell_embed_dim", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--log_every_batches", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--rebuild_cache", action="store_true")
    p.add_argument("--holdout_cell_types_n", type=int, default=3)
    p.add_argument("--holdout_cell_types", default="",
                   help="comma-separated cell_types to hold out (overrides random), "
                        "e.g. 'B1,Tregs,plasma'")
    p.add_argument("--pos_weight_mode", default="auto",
                   help="'auto' (= n_neg/n_pos), 'none', or a float")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no_keep_all_epochs", action="store_true",
                   help="if set, only keep the 3 most recent epoch checkpoints (+ best)")
    p.add_argument("--max_chunks", type=int, default=0,
                   help="if >0, only read the first N chunks (for fast pipeline checks)")
    p.add_argument("--seed_rule", default=DEFAULT_SEED_RULE,
                   choices=sorted(SEED_RULES),
                   help=f"seed-target matching rule (default: {DEFAULT_SEED_RULE})")
    p.add_argument("--cell_encoding", default="cell_type",
                   choices=["cell_type", "hier"],
                   help="'cell_type' = 16 lookup; 'hier' = H1 (cell_type + per-cell_id offset)")
    p.add_argument("--cell_id_weight_decay", type=float, default=1e-2,
                   help="weight_decay for cell_id_embedding (only used when cell_encoding=hier)")
    a = p.parse_args()
    holdout_list = None
    if a.holdout_cell_types.strip():
        holdout_list = [s.strip() for s in a.holdout_cell_types.split(",") if s.strip()]
    return Config(
        run_tag=a.run_tag, model_kind=a.model, batch_size=a.batch_size,
        lr=a.lr, weight_decay=a.weight_decay, max_epochs=a.max_epochs,
        patience=a.patience, cell_embed_dim=a.cell_embed_dim,
        num_workers=a.num_workers, log_every_batches=a.log_every_batches,
        seed=a.seed, device=a.device, rebuild_cache=a.rebuild_cache,
        holdout_cell_types_n=a.holdout_cell_types_n,
        holdout_cell_types=holdout_list,
        pos_weight_mode=a.pos_weight_mode,
        resume=a.resume,
        keep_all_epochs=(not a.no_keep_all_epochs),
        max_chunks=int(a.max_chunks),
        seed_rule=str(a.seed_rule),
        cell_encoding=str(a.cell_encoding),
        cell_id_weight_decay=float(a.cell_id_weight_decay),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = parse_args()
    run_dir = RUNS_DIR / cfg.run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(run_dir / "train_log.txt")
    logger.log(f"[setup] config={json.dumps(cfg.__dict__, indent=2)}")
    logger.log(f"[setup] V3_ROOT={V3_ROOT}")
    logger.log(f"[setup] DATASET_DIR={DATASET_DIR}")
    logger.log(f"[setup] CACHE_DIR={CACHE_DIR}")

    if not DATASET_DIR.exists():
        logger.log(f"[error] dataset directory not found: {DATASET_DIR}")
        sys.exit(1)

    np.random.seed(cfg.seed)

    # ---- Phase 1: build cache (or reuse) ----
    if cfg.rebuild_cache and CACHE_DIR.exists():
        logger.log(f"[phase1] --rebuild_cache: removing {CACHE_DIR}")
        shutil.rmtree(CACHE_DIR)

    if not cache_exists(CACHE_DIR, expected_seed_rule=cfg.seed_rule):
        logger.log(f"[phase1] building cache from scratch (seed_rule={cfg.seed_rule})")
        chunks = discover_chunks(DATASET_DIR,
                                 max_chunks=cfg.max_chunks if cfg.max_chunks > 0 else None)
        logger.log(f"[phase1] using {len(chunks)} chunks (max_chunks={cfg.max_chunks})")
        if not chunks:
            logger.log("[error] no chunks found")
            sys.exit(1)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        gmaps = build_global_maps(DATASET_DIR, chunks, logger)
        master = build_master_samples(DATASET_DIR, chunks, gmaps, logger)
        split_cfg = SplitConfig(
            holdout_cell_types_n=cfg.holdout_cell_types_n,
            holdout_cell_types=cfg.holdout_cell_types,
            seed=cfg.seed,
        )
        splits = make_paper_split(
            master, split_cfg, gmaps["cell_type_to_idx"], logger,
            cell_id_to_idx=gmaps.get("cell_id_to_idx"),
        )
        del master
        write_cache(gmaps, splits, CACHE_DIR, logger)
        encode_global_dictionaries(gmaps, splits, CACHE_DIR, logger, cfg.seed_rule)
        del gmaps, splits
    else:
        logger.log(f"[phase1] reusing existing cache (seed_rule={cfg.seed_rule}; "
                   f"pass --rebuild_cache to force rebuild)")

    # ---- Early exit: cache-build-only mode (max_epochs == 0) ----
    if cfg.max_epochs == 0:
        logger.log("[stop] max_epochs=0 -> cache-build-only mode; exiting before training")
        logger.close()
        return

    # ---- Phase 2: load cache ----
    cache = load_cache(CACHE_DIR, logger)
    cell_type_to_idx: dict[str, int] = cache["cell_type_to_idx"]
    idx_to_cell_type = {int(v): k for k, v in cell_type_to_idx.items()}
    n_cell_types = len(cell_type_to_idx)
    logger.log(f"[setup] n_cell_types={n_cell_types}")

    # echo seed-anchor report from cache
    rep_path = CACHE_DIR / "seed_anchor_report.json"
    if rep_path.exists():
        rep = json.loads(rep_path.read_text(encoding="utf-8"))
        logger.log(f">>> SEED ANCHOR REPORT (from cache): pairs={rep['n_unique_pairs']:,} "
                   f"hit={rep['n_seed_anchor_found']:,} fallback={rep['n_seed_anchor_fallback']:,} "
                   f"hit_rate={rep['seed_anchor_hit_rate']:.4f} <<<")

    # ---- Phase 3: torch setup ----
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)
    logger.log(f"[setup] device={device}")

    def _make_ds(name: str):
        return LazyMTIDataset(
            rows=cache["splits"][name],
            mirna_codes_global=cache["mirna_codes_global"],
            seed_targets=cache["seed_targets"],
            utr_seqs=cache["utr_seqs"],
        )

    train_ds = _make_ds("train")
    val_ds = _make_ds("val")
    eval_names = ["test_triple", "test_cell_holdout", "test_pair_x_cell_holdout"]
    eval_dss: dict[str, LazyMTIDataset] = {}
    for n in eval_names:
        if n in cache["splits"]:
            eval_dss[n] = _make_ds(n)
    logger.log(
        f"[data] train={len(train_ds):,} val={len(val_ds):,} "
        + " ".join(f"{n}={len(ds):,}" for n, ds in eval_dss.items())
    )

    # ---- Phase 4: model + optimiser + loss ----
    n_cell_ids = len(cache["cell_id_to_idx"]) if cache.get("cell_id_to_idx") else 0
    if cfg.cell_encoding == "hier" and n_cell_ids == 0:
        logger.log("[error] cell_encoding='hier' requested but cache has no cell_id_to_idx; "
                   "rebuild cache with current code (it will write cell_id_to_idx.json).")
        sys.exit(1)
    model = make_model(
        n_cell_types, cfg.model_kind, cfg.cell_embed_dim,
        cell_encoding=cfg.cell_encoding, n_cell_ids=n_cell_ids,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"[model] kind={cfg.model_kind} cell_encoding={cfg.cell_encoding} "
               f"n_cell_ids={n_cell_ids} trainable_params={n_params:,}")

    # pos_weight from train labels
    train_y = cache["splits"]["train"]["label"]
    n_pos = int((train_y == 1).sum())
    n_neg = int((train_y == 0).sum())
    if cfg.pos_weight_mode == "auto":
        pw_val = n_neg / max(n_pos, 1)
        pos_weight_tensor = torch.tensor([pw_val], dtype=torch.float32, device=device)
        logger.log(f"[loss] auto pos_weight={pw_val:.4f} (n_neg={n_neg:,} n_pos={n_pos:,})")
    elif cfg.pos_weight_mode == "none":
        pos_weight_tensor = None
        logger.log("[loss] pos_weight disabled")
    else:
        try:
            pw_val = float(cfg.pos_weight_mode)
        except ValueError:
            logger.log(f"[error] invalid pos_weight_mode: {cfg.pos_weight_mode}")
            sys.exit(1)
        pos_weight_tensor = torch.tensor([pw_val], dtype=torch.float32, device=device)
        logger.log(f"[loss] manual pos_weight={pw_val:.4f}")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # H1: cell_id_emb gets a larger weight_decay to fight overfit (only ~80 rows/cell_id).
    if cfg.cell_encoding == "hier" and any("cell_id_embedding" in n for n, _ in model.named_parameters()):
        main_params = [p for n, p in model.named_parameters() if "cell_id_embedding" not in n]
        cid_params  = [p for n, p in model.named_parameters() if "cell_id_embedding" in n]
        opt = torch.optim.AdamW([
            {"params": main_params, "weight_decay": cfg.weight_decay},
            {"params": cid_params,  "weight_decay": cfg.cell_id_weight_decay},
        ], lr=cfg.lr)
        logger.log(f"[opt] AdamW with grouped wd: main={cfg.weight_decay} "
                   f"cell_id={cfg.cell_id_weight_decay}")
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=1)

    pin = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, collate_fn=collate_batch,
                              pin_memory=pin,
                              persistent_workers=(cfg.num_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size * 2, shuffle=False,
                            num_workers=cfg.num_workers, collate_fn=collate_batch,
                            pin_memory=pin,
                            persistent_workers=(cfg.num_workers > 0))

    # ---- Phase 5: training loop with resume support and per-epoch checkpoints ----
    history: list[dict] = []
    best_pr_auc = -math.inf
    best_epoch = -1
    epochs_no_improve = 0
    start_epoch = 1

    if cfg.resume:
        ckpts = sorted(run_dir.glob("checkpoint_epoch_*.pt"))
        if ckpts:
            def _ep(p: Path) -> int:
                try:
                    return int(p.stem.split("_")[-1])
                except Exception:
                    return -1
            latest = sorted(ckpts, key=_ep)[-1]
            logger.log(f"[resume] loading checkpoint from {latest.name}")
            ckpt = torch.load(latest, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            opt.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = int(ckpt["epoch"]) + 1
            history = list(ckpt.get("history", []))
            best_pr_auc = float(ckpt.get("best_pr_auc", -math.inf))
            best_epoch = int(ckpt.get("best_epoch", -1))
            epochs_no_improve = int(ckpt.get("epochs_no_improve", 0))
            try:
                torch_rng = ckpt.get("torch_rng_state")
                if torch_rng is not None:
                    torch.set_rng_state(torch_rng)
                np_rng = ckpt.get("numpy_rng_state")
                if np_rng is not None:
                    np.random.set_state(np_rng)
            except Exception as e:
                logger.log(f"[resume] could not restore RNG state: {e}")
            logger.log(
                f"[resume] resumed at epoch {start_epoch} "
                f"(best so far: epoch {best_epoch} PR-AUC={best_pr_auc:.4f}, "
                f"epochs_no_improve={epochs_no_improve})"
            )
        else:
            logger.log("[resume] --resume given but no checkpoint_epoch_*.pt found; starting fresh")

    if start_epoch > cfg.max_epochs:
        logger.log(f"[stop] start_epoch={start_epoch} > max_epochs={cfg.max_epochs}; "
                   f"skipping training, going straight to final eval")

    for epoch in range(start_epoch, cfg.max_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, loss_fn, opt, device,
                                     logger, cfg.log_every_batches, epoch)
        val = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss_mean": val["val_loss_mean"],
            "val_pr_auc": val["pr_auc"],
            "val_roc_auc": val["roc_auc"],
            "val_best_f1": val["best_f1"],
            "val_best_threshold": val["best_threshold"],
            "val_p_at_top1pct": val["p_at_top_1pct"],
            "val_p_at_top5pct": val["p_at_top_5pct"],
            "val_p_at_top10pct": val["p_at_top_10pct"],
            "elapsed_s": elapsed,
        }
        history.append(row)
        logger.log(
            f"[epoch {epoch}] train_loss={train_loss:.4f} val_loss={val['val_loss_mean']:.4f} "
            f"PR-AUC={val['pr_auc']:.4f} ROC-AUC={val['roc_auc']:.4f} "
            f"F1={val['best_f1']:.4f}@{val['best_threshold']:.2f} "
            f"P@1%={val['p_at_top_1pct']:.3f} P@5%={val['p_at_top_5pct']:.3f} "
            f"elapsed={elapsed:.1f}s"
        )

        improved = val["pr_auc"] > best_pr_auc + 1e-6
        if improved:
            best_pr_auc = val["pr_auc"]
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # full per-epoch checkpoint
        ckpt_path = run_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_metrics": {k: v for k, v in val.items() if not k.startswith("_")},
            "history": history,
            "best_pr_auc": best_pr_auc,
            "best_epoch": best_epoch,
            "epochs_no_improve": epochs_no_improve,
            "config": cfg.__dict__,
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
        }, ckpt_path)

        # last (lightweight)
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_metrics": {k: v for k, v in val.items() if not k.startswith("_")},
            "config": cfg.__dict__,
        }, run_dir / "last_model.pt")

        if improved:
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_metrics": {k: v for k, v in val.items() if not k.startswith("_")},
                "config": cfg.__dict__,
            }, run_dir / "best_model.pt")
            logger.log(f"[best] new best at epoch {epoch} PR-AUC={best_pr_auc:.4f}")
        else:
            logger.log(f"[best] no improve ({epochs_no_improve}/{cfg.patience})")

        if not cfg.keep_all_epochs:
            all_ckpts = sorted(run_dir.glob("checkpoint_epoch_*.pt"))
            if len(all_ckpts) > 3:
                for old in all_ckpts[:-3]:
                    try:
                        ep_num = int(old.stem.split("_")[-1])
                    except Exception:
                        ep_num = -1
                    if ep_num != best_epoch:
                        try:
                            old.unlink()
                        except OSError:
                            pass

        scheduler.step(val["pr_auc"])
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        if epochs_no_improve >= cfg.patience:
            logger.log(f"[stop] early stopping at epoch {epoch}")
            break

    # ---- Phase 6: final evaluation on test sets ----
    if (run_dir / "best_model.pt").exists():
        logger.log(f"[final] loading best model from epoch {best_epoch} "
                   f"(val PR-AUC={best_pr_auc:.4f})")
        ckpt = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        logger.log("[final] no best_model.pt; using current model weights for final eval")

    final_report = {
        "config": cfg.__dict__,
        "best_epoch": best_epoch,
        "best_val_pr_auc": best_pr_auc,
        "history": history,
        "evaluations": {},
    }

    val_final = evaluate(model, val_loader, device)
    final_report["evaluations"]["val"] = {k: v for k, v in val_final.items() if not k.startswith("_")}
    final_report["evaluations"]["val"]["per_cell"] = per_cell_metrics(val_final, idx_to_cell_type)
    np.savez(run_dir / "val_predictions.npz",
             y_true=val_final["_y_true"], y_score=val_final["_y_score"], cell=val_final["_cell"])

    for name, ds in eval_dss.items():
        if len(ds) == 0:
            logger.log(f"[{name}] empty, skipping")
            continue
        loader = DataLoader(ds, batch_size=cfg.batch_size * 2,
                            shuffle=False, num_workers=cfg.num_workers,
                            collate_fn=collate_batch, pin_memory=pin)
        ev = evaluate(model, loader, device)
        final_report["evaluations"][name] = {k: v for k, v in ev.items() if not k.startswith("_")}
        final_report["evaluations"][name]["per_cell"] = per_cell_metrics(ev, idx_to_cell_type)
        np.savez(run_dir / f"{name}_predictions.npz",
                 y_true=ev["_y_true"], y_score=ev["_y_score"], cell=ev["_cell"])
        logger.log(
            f"[{name}] PR-AUC={ev['pr_auc']:.4f} ROC-AUC={ev['roc_auc']:.4f} "
            f"F1={ev['best_f1']:.4f}@{ev['best_threshold']:.2f} "
            f"P@1%={ev['p_at_top_1pct']:.3f} pos_rate={ev['positive_rate']:.4f} "
            f"n={ev['n']:,}"
        )

    (run_dir / "final_report.json").write_text(
        json.dumps(final_report, indent=2, default=str), encoding="utf-8")
    logger.log(f"[done] final report written to {run_dir / 'final_report.json'}")
    logger.close()


if __name__ == "__main__":
    main()
