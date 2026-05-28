#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V4 — Cell-type aware MTI prediction with stronger UTR encoder.

This file is a copy of train_main_model.py (V3) with the following changes,
made in response to V3's failure to generalise to new UTR sequences
(test_utr_holdout PR-AUC ~ baseline) AND V3-mixed-only diagnostic also
showing no improvement, which together rule out "the model just doesn't use
cell info" and pin the bottleneck to the UTR sequence encoder itself.

V4 changes (relative to V3):

  (1) UTR encoder: vanilla 3-layer Conv1D -> ConvNeXt-1D, 4 stages.
      (REPRESS Fig g: ConvNeXt-style beats Mamba +10% and dilated CNN +17%
       on similar miRNA tasks at the same parameter budget.)
      ~140K trainable params -> ~1.6M trainable params (~11x scale-up).

  (2) UTR window: 2000 nt -> 4000 nt (still seed-anchored).
      Covers ~75% of UTRs end-to-end (median UTR length is 2,674 nt).
      Captures multiple seed sites and longer RBP context.

  (3) Auxiliary task: per-position seed-match prediction head.
      A second classifier head on the UTR encoder output predicts, for each
      position in the (downsampled) UTR window, P(this position is a
      seed-target match for this miRNA). Ground-truth labels are built on the
      fly from the same find-all-seed-matches logic used for cropping.
      Trained jointly with the main repression task; total loss is
            L = L_repression + aux_weight * L_seed_match    (aux_weight=0.3)
      Rationale: V3-mixed showed the UTR encoder does not learn transferable
      seed-level features. Direct supervision on "where is the seed?" forces
      it to.

  (4) Cache reuse: V4 shares the V3 cache transparently. Seed-position masks
      are computed on the fly inside the dataset, no cache rebuild needed.

Everything else (12-set holdout, FiLM cell injection, miRNA-conditioned
attention, TargetScan-style hand-crafted features, Focal loss + dual sample
weights, normalised-lift early stop on val_utr_holdout) is unchanged from V3.

Run (after the V3 cache exists):

    python script/train_main_model_v4.py \\
        --run_tag v4_film_seed42 \\
        --model film \\
        --batch_size 128 --num_workers 8 --device auto \\
        --max_epochs 12 --patience 3 \\
        --holdout_cell_types "B1,Tregs,plasma,mDC1" \\
        --val_holdout_cell_types "mDC1" \\
        --seed_rule any_7mer

(or use the helper:  ./script/run_v4.sh)

============================ original V3 header below =========================

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
V3_ROOT = SCRIPT_DIR.parent
DATASET_DIR = V3_ROOT / "dataset"
CACHE_DIR = V3_ROOT / "cache"
RUNS_DIR = V3_ROOT / "runs"


# ---------------------------------------------------------------------------
# Constants for sequence encoding
# ---------------------------------------------------------------------------

MIRNA_LEN = 26                  # pad miRNA seqs (which are 18-24 nt) to 26
UTR_WINDOW_LEN = 4000           # V4: doubled from 2000 to cover ~75% of UTRs end-to-end
UTR_DOWNSAMPLE_V4 = 16          # ConvNeXt stem(2) x stages(2,2,2) = 16x downsample; 4000/16 = 250
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


def find_all_seed_matches(utr_seq: str, seed_target) -> list[int]:
    """V4: Find ALL (non-overlapping) seed-target match positions in utr_seq.

    seed_target may be a single string or a list of candidate motifs (multi-rule).
    Returns a sorted list of 0-indexed positions where any motif begins.

    Overlapping matches of different motifs at the same position are kept once
    (de-duplicated). Used by the auxiliary seed-mask supervision in V4.
    """
    if seed_target is None:
        return []
    targets = [seed_target] if isinstance(seed_target, str) else list(seed_target)
    if not targets:
        return []
    utr_norm = utr_seq.upper().replace("U", "T")
    positions: set[int] = set()
    for t in targets:
        if not t:
            continue
        t_norm = t.upper().replace("U", "T")
        L = len(t_norm)
        i = 0
        while True:
            j = utr_norm.find(t_norm, i)
            if j < 0:
                break
            positions.add(j)
            i = j + 1  # allow overlap with next motif start, like TargetScan
    return sorted(positions)


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


def compute_targetscan_features(utr_seq: str, seed_targets, au_radius: int = 50) -> np.ndarray:
    """5-dim TargetScan-style hand-crafted features for a (mirna, utr) pair.

    Returns a float32 array of length 5:
      [0] has_seed_match      : 1.0 if any seed motif matches in UTR else 0.0
      [1] seed_pos_normalized : anchor_pos / max(L-1, 1) in [0, 1]; 0.0 if no match
      [2] AU_content_around   : (A + T) fraction in ±au_radius window around anchor;
                                 if no match, computed over full UTR (fallback)
      [3] log_num_seed_matches: log1p of total seed-motif occurrences (counted via
                                 str.count for each motif — overlapping counts OK,
                                 this is a coarse signal)
      [4] utr_length_log      : log(L + 1)

    These are based on TargetScan-style canonical features and are computed once per
    unique (utr_gid, mirna_gid) pair so they cost nothing at train-step time.
    """
    feats = np.zeros(5, dtype=np.float32)
    if not utr_seq:
        return feats
    L = len(utr_seq)
    utr_norm = utr_seq.upper().replace("U", "T")
    feats[4] = float(np.log(L + 1))

    targets = [seed_targets] if isinstance(seed_targets, str) else list(seed_targets or [])
    if not targets:
        # No seed → AU over full UTR, length only
        n_au = utr_norm.count("A") + utr_norm.count("T")
        feats[2] = n_au / max(L, 1)
        return feats

    # find earliest match across all motifs
    anchor = -1
    total_hits = 0
    for t in targets:
        if not t:
            continue
        t_norm = t.upper().replace("U", "T")
        pos = utr_norm.find(t_norm)
        if pos >= 0 and (anchor < 0 or pos < anchor):
            anchor = pos
        total_hits += utr_norm.count(t_norm)

    if anchor < 0:
        # No seed match → AU over full UTR as fallback
        n_au = utr_norm.count("A") + utr_norm.count("T")
        feats[0] = 0.0
        feats[1] = 0.0
        feats[2] = n_au / max(L, 1)
        feats[3] = 0.0  # log1p(0) = 0
        return feats

    feats[0] = 1.0
    feats[1] = float(anchor) / max(L - 1, 1)
    w_start = max(0, anchor - au_radius)
    w_end = min(L, anchor + au_radius)
    window = utr_norm[w_start:w_end]
    if window:
        n_au = window.count("A") + window.count("T")
        feats[2] = n_au / len(window)
    feats[3] = float(np.log1p(total_hits))
    return feats


TARGETSCAN_FEATURE_DIM = 5


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
    logger.log(f"[global_map] TOTAL unique miRNAs={len(mirna_seqs)} "
               f"UTRs={len(utr_seqs)} cells={len(cell_id_to_type)} "
               f"cell_types={len(cell_type_to_idx)}")
    logger.log(f"[global_map] cell_types: {all_cell_types}")

    return {
        "mirna_seqs": mirna_seqs,
        "utr_seqs": utr_seqs,
        "cell_id_to_type": cell_id_to_type,
        "cell_type_to_idx": cell_type_to_idx,
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
    utr_holdout_frac: float = 0.20      # 20% of UTR sequences held out entirely
    mirna_holdout_frac: float = 0.20    # 20% of miRNA sequences held out entirely
    val_utr_frac_of_holdout: float = 0.30  # fraction of UTR holdout used as val_utr_holdout for early stopping
    val_holdout_cell_types: list[str] | None = None  # subset of holdout_cell_types reserved for val_cell_holdout (early stopping); rest go to test_cell_holdout
    train_frac: float = 0.70
    val_frac: float = 0.10
    test_frac: float = 0.20
    seed: int = 42


def _stratified_holdout(values: list, pos_rates: dict, frac: float, seed: int,
                        bins: tuple = (0.1, 0.3, 0.7, 0.9)) -> set:
    """Select holdout members stratified by pos_rate bins.

    Splits all values into bins (e.g. near-negative / weak-negative / balanced /
    weak-positive / near-positive) and samples `frac` from each bin so the
    holdout has the same pos_rate distribution as the kept set. Stable across
    runs via _stable_seed.
    """
    bins_arr = sorted(set(list(bins) + [0.0, 1.0]))
    def _bin(p):
        for i in range(len(bins_arr) - 1):
            if bins_arr[i] <= p < bins_arr[i + 1]:
                return i
        return len(bins_arr) - 2
    by_bin: dict[int, list] = {}
    for v in values:
        by_bin.setdefault(_bin(pos_rates[v]), []).append(v)
    holdout = set()
    for b, items in by_bin.items():
        n = int(round(len(items) * frac))
        if n == 0:
            continue
        items_sorted = sorted(items, key=lambda x: _stable_seed(seed, "holdout", b, x))
        holdout.update(items_sorted[:n])
    return holdout


def make_paper_split(
    master: dict[str, np.ndarray],
    cfg: SplitConfig,
    cell_type_to_idx: dict[str, int],
    logger: Logger,
) -> dict[str, Any]:
    """Three-axis holdout split with TWELVE evaluation sets:

      train                          !UTR_h, !mRNA_h, !cell_h,  pair=train
      val                            !UTR_h, !mRNA_h, !cell_h,  pair=val
      test_triple                    !UTR_h, !mRNA_h, !cell_h,  pair=test
      val_cell_holdout               !UTR_h, !mRNA_h,  vcell_h, all pairs       (early stop)
      test_cell_holdout              !UTR_h, !mRNA_h,  tcell_h, all pairs
      test_pair_x_cell_holdout       !UTR_h, !mRNA_h,  tcell_h, pair=test
      val_utr_holdout                 vUTR_h,!mRNA_h, !cell_h,  all rows         (early stop)
      val_utr_x_cell_holdout          vUTR_h,!mRNA_h,  vcell_h, all rows         (early stop, hardest val)
      test_utr_holdout                tUTR_h,!mRNA_h, !cell_h,  all rows
      test_utr_x_cell_holdout         tUTR_h,!mRNA_h,  tcell_h, all rows
      test_mirna_holdout             !UTR_h,  mRNA_h, !cell_h,  all rows
      test_mirna_x_cell_holdout      !UTR_h,  mRNA_h,  tcell_h, all rows

    Pipeline:
      Step 1: holdout cell_types (manual or random, ~3 types).
      Step 2: holdout UTRs (~20%, stratified by per-UTR pos_rate).
      Step 3: holdout miRNAs (~20%, stratified by per-miRNA pos_rate).
      Step 4: both-label filter on (mirna_gid, cell_id) groups, applied
              independently to each (UTR_holdout, mRNA_holdout, cell_holdout)
              subspace so each evaluation set is non-trivial.
      Step 5: pair-level partition (train/val/test) applied ONLY within the
              (!UTR_h, !mRNA_h) subspace. Holdout-UTR/-mRNA rows are not
              pair-bucketed since their pairs are intrinsically new.
      Step 6: emit the 9 sets by intersecting holdout axes with pair bucket.

    Strict invariants verified by diagnose_leakage.py:
      - train has zero (mirna, utr) pair overlap with val / test_triple /
        test_pair_x_cell_holdout
      - train has zero UTR overlap with test_utr_holdout / test_utr_x_cell_holdout
      - train has zero miRNA overlap with test_mirna_holdout / test_mirna_x_cell_holdout
    """
    cell_id_arr = master["cell_id"]
    cell_type_arr = master["cell_type"]
    mirna_gid = master["mirna_gid"]
    utr_gid = master["utr_gid"]
    label = master["label"]

    n_orig = len(label)
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

    # ---- Step 1: cell-type holdout ----
    if cfg.holdout_cell_types is not None:
        requested = set(cfg.holdout_cell_types)
        unknown = requested - set(all_cell_types)
        if unknown:
            raise ValueError(f"unknown cell_types in --holdout_cell_types: {sorted(unknown)}")
        holdout_cts = sorted(requested)
        logger.log(f"[split] MANUAL cell holdout ({len(holdout_cts)}): {holdout_cts}")
    else:
        n_hold = min(cfg.holdout_cell_types_n, max(0, len(all_cell_types) - 1))
        perm = rng.permutation(len(all_cell_types))
        holdout_cts = sorted([all_cell_types[i] for i in perm[:n_hold]])
        logger.log(f"[split] RANDOM cell holdout ({len(holdout_cts)}): {holdout_cts}")
    holdout_ct_set = set(holdout_cts)
    # Further split holdout_cell_types into val_cell (early stopping) and test_cell (final eval)
    if cfg.val_holdout_cell_types is not None:
        val_cell_set = set(cfg.val_holdout_cell_types) & holdout_ct_set
    elif len(holdout_cts) >= 2:
        # default: pick 1 cell type as val (deterministic by stable hash)
        ranked = sorted(holdout_cts, key=lambda c: _stable_seed(cfg.seed, "val_cell", c))
        val_cell_set = {ranked[0]}
    else:
        val_cell_set = set()
    test_cell_set = holdout_ct_set - val_cell_set
    cell_test_h_mask = np.array([ct in test_cell_set for ct in cell_type_arr])
    cell_val_h_mask  = np.array([ct in val_cell_set  for ct in cell_type_arr])
    cell_main_mask   = ~(cell_test_h_mask | cell_val_h_mask)
    logger.log(f"[split] cell groups: main={int(cell_main_mask.sum()):,} rows ({sorted(set(all_cell_types)-holdout_ct_set)}), "
               f"val_cell={int(cell_val_h_mask.sum()):,} rows ({sorted(val_cell_set)}), "
               f"test_cell={int(cell_test_h_mask.sum()):,} rows ({sorted(test_cell_set)})")

    # ---- Step 2: UTR holdout (stratified by per-UTR pos_rate) ----
    utr_pos_sum: dict[int, float] = {}
    utr_count: dict[int, int] = {}
    for u, y in zip(utr_gid.tolist(), label.tolist()):
        utr_count[u] = utr_count.get(u, 0) + 1
        utr_pos_sum[u] = utr_pos_sum.get(u, 0.0) + float(y)
    utr_pos_rate = {u: utr_pos_sum[u] / utr_count[u] for u in utr_count}
    unique_utrs = sorted(utr_pos_rate.keys())
    utr_h_set = _stratified_holdout(unique_utrs, utr_pos_rate,
                                    cfg.utr_holdout_frac, cfg.seed + 1) \
                if cfg.utr_holdout_frac > 0 else set()
    utr_h_mask = np.array([int(u) in utr_h_set for u in utr_gid])
    # further split utr_h_set into val_utr_h (for early stopping) and test_utr_h (for final eval)
    utr_h_sorted = sorted(utr_h_set, key=lambda x: _stable_seed(cfg.seed, "utr_valtest", x))
    n_val_utr = int(round(len(utr_h_sorted) * cfg.val_utr_frac_of_holdout))
    val_utr_h_set = set(utr_h_sorted[:n_val_utr])
    test_utr_h_set = set(utr_h_sorted[n_val_utr:])
    val_utr_h_mask = np.array([int(u) in val_utr_h_set for u in utr_gid])
    test_utr_h_mask = np.array([int(u) in test_utr_h_set for u in utr_gid])
    logger.log(f"[split] UTR holdout total: {len(utr_h_set):,} / {len(unique_utrs):,} UTRs "
               f"({len(utr_h_set)/max(len(unique_utrs),1):.1%})")
    logger.log(f"[split]   --> val_utr_h: {len(val_utr_h_set):,} UTRs "
               f"({int(val_utr_h_mask.sum()):,} rows; for early stopping)")
    logger.log(f"[split]   --> test_utr_h: {len(test_utr_h_set):,} UTRs "
               f"({int(test_utr_h_mask.sum()):,} rows; for final eval)")

    # ---- Step 3: miRNA holdout (stratified by per-mirna pos_rate) ----
    mirna_pos_sum: dict[int, float] = {}
    mirna_count: dict[int, int] = {}
    for m, y in zip(mirna_gid.tolist(), label.tolist()):
        mirna_count[m] = mirna_count.get(m, 0) + 1
        mirna_pos_sum[m] = mirna_pos_sum.get(m, 0.0) + float(y)
    mirna_pos_rate = {m: mirna_pos_sum[m] / mirna_count[m] for m in mirna_count}
    unique_mirnas = sorted(mirna_pos_rate.keys())
    mirna_h_set = _stratified_holdout(unique_mirnas, mirna_pos_rate,
                                       cfg.mirna_holdout_frac, cfg.seed + 2) \
                  if cfg.mirna_holdout_frac > 0 else set()
    mirna_h_mask = np.array([int(m) in mirna_h_set for m in mirna_gid])
    logger.log(f"[split] miRNA holdout: {len(mirna_h_set):,} / {len(unique_mirnas):,} miRNAs "
               f"({len(mirna_h_set)/max(len(unique_mirnas),1):.1%}), "
               f"affecting {int(mirna_h_mask.sum()):,} rows")

    # ---- Subspace masks (utr × mirna × cell) for 12 sets ----
    # UTR has 3 categories: main (~), val_utr_h, test_utr_h
    # cell has 3 categories: main, val_cell_h, test_cell_h
    # mirna has 2 categories: main (~), mirna_h
    # We name s_<utr>_<mirna>_<cell> where:
    #   utr:  o=main, v=val_h, t=test_h
    #   mirna: o=main, x=held out
    #   cell: o=main, v=val_h, t=test_h
    utr_main_mask = ~utr_h_mask
    # 12 used subspaces (others dropped to avoid set explosion)
    s_main         = utr_main_mask     & (~mirna_h_mask) & cell_main_mask    # train/val/test_triple
    s_o_o_tcell    = utr_main_mask     & (~mirna_h_mask) & cell_test_h_mask  # test_cell_holdout family
    s_o_o_vcell    = utr_main_mask     & (~mirna_h_mask) & cell_val_h_mask   # val_cell_holdout (early stopping)
    s_vutr_o_omain = val_utr_h_mask    & (~mirna_h_mask) & cell_main_mask    # val_utr_holdout (early stopping)
    s_vutr_o_vcell = val_utr_h_mask    & (~mirna_h_mask) & cell_val_h_mask   # val_utr_x_cell_holdout (early stop, mirrors test_utr_x_cell)
    s_tutr_o_omain = test_utr_h_mask   & (~mirna_h_mask) & cell_main_mask    # test_utr_holdout
    s_tutr_o_tcell = test_utr_h_mask   & (~mirna_h_mask) & cell_test_h_mask  # test_utr_x_cell_holdout
    s_o_mh_omain   = utr_main_mask     & ( mirna_h_mask) & cell_main_mask    # test_mirna_holdout
    s_o_mh_tcell   = utr_main_mask     & ( mirna_h_mask) & cell_test_h_mask  # test_mirna_x_cell_holdout

    # ---- Step 4: both-label filter per subspace ----
    def _both_label_keep(rows_mask: np.ndarray) -> tuple[np.ndarray, int, int]:
        if not rows_mask.any():
            return rows_mask.copy(), 0, 0
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
        full_keep = np.zeros(len(rows_mask), dtype=bool)
        sub_indices = np.nonzero(rows_mask)[0]
        full_keep[sub_indices[local_keep]] = True
        return full_keep, len(group_pn), len(kept)

    k_main,        ng_main,   nk_main   = _both_label_keep(s_main)
    k_o_o_tcell,   ng_tcell,  nk_tcell  = _both_label_keep(s_o_o_tcell)
    k_o_o_vcell,   ng_vcell,  nk_vcell  = _both_label_keep(s_o_o_vcell)
    k_vutr,        ng_vutr,   nk_vutr   = _both_label_keep(s_vutr_o_omain)
    k_vutr_vcell,  ng_vuxc,   nk_vuxc   = _both_label_keep(s_vutr_o_vcell)
    k_tutr,        ng_tutr,   nk_tutr   = _both_label_keep(s_tutr_o_omain)
    k_tutr_tcell,  ng_uxc,    nk_uxc    = _both_label_keep(s_tutr_o_tcell)
    k_mh,          ng_mh,     nk_mh     = _both_label_keep(s_o_mh_omain)
    k_mh_tcell,    ng_mhxc,   nk_mhxc   = _both_label_keep(s_o_mh_tcell)
    logger.log(f"[split] both-label group survival: "
               f"main={nk_main}/{ng_main} "
               f"test_cell={nk_tcell}/{ng_tcell} val_cell={nk_vcell}/{ng_vcell} "
               f"val_utr={nk_vutr}/{ng_vutr} val_utrXcell={nk_vuxc}/{ng_vuxc} "
               f"test_utr={nk_tutr}/{ng_tutr} utrXcell={nk_uxc}/{ng_uxc} "
               f"mirnaHold={nk_mh}/{ng_mh} mirnaXcell={nk_mhxc}/{ng_mhxc}")

    # ---- Step 5: pair-level partition (only within main_UTR × main_miRNA × any cell) ----
    train_thr = cfg.train_frac
    val_thr = cfg.train_frac + cfg.val_frac
    # Pair partition applies to (main UTR, main miRNA) pairs, which appear in s_main + s_o_o_tcell + s_o_o_vcell
    pair_eligible_mask = k_main | k_o_o_tcell | k_o_o_vcell
    unique_pairs = set(zip(mirna_gid[pair_eligible_mask].tolist(),
                           utr_gid[pair_eligible_mask].tolist()))
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
    n_pair_train = sum(1 for b in pair_buckets if b == 0)
    n_pair_val   = sum(1 for b in pair_buckets if b == 1)
    n_pair_test  = sum(1 for b in pair_buckets if b == 2)
    logger.log(f"[split] unique main-UTR/main-miRNA pairs: {n_unique_pairs:,} "
               f"(train={n_pair_train:,} val={n_pair_val:,} test={n_pair_test:,})")

    def _row_buckets(rows_mask: np.ndarray) -> np.ndarray:
        if not rows_mask.any():
            return np.array([], dtype=np.int8)
        sub_m = mirna_gid[rows_mask].tolist()
        sub_u = utr_gid[rows_mask].tolist()
        return np.fromiter(
            (pair_to_bucket.get((int(m), int(u)), -1) for m, u in zip(sub_m, sub_u)),
            dtype=np.int8, count=int(rows_mask.sum()),
        )

    b_main = _row_buckets(k_main)
    b_tcell = _row_buckets(k_o_o_tcell)

    # ---- Step 6: assemble the 11 sets ----
    def _pack(rows_mask: np.ndarray, local_select: np.ndarray) -> dict[str, np.ndarray]:
        if (not rows_mask.any()) or (not local_select.any()):
            return {
                "mirna_gid": np.zeros(0, dtype=np.int32),
                "utr_gid":   np.zeros(0, dtype=np.int32),
                "cell_idx":  np.zeros(0, dtype=np.int32),
                "cell_type": np.zeros(0, dtype=object),
                "cell_id":   np.zeros(0, dtype=object),
                "label":     np.zeros(0, dtype=np.uint8),
            }
        sub_indices = np.nonzero(rows_mask)[0][local_select]
        ct = cell_type_arr[sub_indices]
        return {
            "mirna_gid": mirna_gid[sub_indices].astype(np.int32),
            "utr_gid":   utr_gid[sub_indices].astype(np.int32),
            "cell_idx":  np.array([cell_type_to_idx[c] for c in ct], dtype=np.int32),
            "cell_type": ct.astype(object),
            "cell_id":   cell_id_arr[sub_indices].astype(object),
            "label":     label[sub_indices].astype(np.uint8),
        }

    train_d                     = _pack(k_main, b_main == 0)
    val_d                       = _pack(k_main, b_main == 1)
    test_triple_d               = _pack(k_main, b_main == 2)
    test_cell_holdout_d         = _pack(k_o_o_tcell, np.ones(int(k_o_o_tcell.sum()), dtype=bool))
    test_pair_x_cell_holdout_d  = _pack(k_o_o_tcell, b_tcell == 2)
    val_cell_holdout_d          = _pack(k_o_o_vcell, np.ones(int(k_o_o_vcell.sum()), dtype=bool))
    val_utr_holdout_d           = _pack(k_vutr, np.ones(int(k_vutr.sum()), dtype=bool))
    val_utr_x_cell_holdout_d    = _pack(k_vutr_vcell, np.ones(int(k_vutr_vcell.sum()), dtype=bool))
    test_utr_holdout_d          = _pack(k_tutr, np.ones(int(k_tutr.sum()), dtype=bool))
    test_utr_x_cell_holdout_d   = _pack(k_tutr_tcell, np.ones(int(k_tutr_tcell.sum()), dtype=bool))
    test_mirna_holdout_d        = _pack(k_mh, np.ones(int(k_mh.sum()), dtype=bool))
    test_mirna_x_cell_holdout_d = _pack(k_mh_tcell, np.ones(int(k_mh_tcell.sum()), dtype=bool))

    def _pr(d):
        return float(d["label"].mean()) if len(d["label"]) else 0.0

    splits_meta: dict[str, dict] = {}
    for name, d in [
        ("train", train_d),
        ("val", val_d),
        ("test_triple", test_triple_d),
        ("val_cell_holdout", val_cell_holdout_d),
        ("test_cell_holdout", test_cell_holdout_d),
        ("test_pair_x_cell_holdout", test_pair_x_cell_holdout_d),
        ("val_utr_holdout", val_utr_holdout_d),
        ("val_utr_x_cell_holdout", val_utr_x_cell_holdout_d),
        ("test_utr_holdout", test_utr_holdout_d),
        ("test_utr_x_cell_holdout", test_utr_x_cell_holdout_d),
        ("test_mirna_holdout", test_mirna_holdout_d),
        ("test_mirna_x_cell_holdout", test_mirna_x_cell_holdout_d),
    ]:
        splits_meta[name] = {"n": int(len(d["label"])), "positive_rate": _pr(d)}
        logger.log(f"[split] {name}: n={splits_meta[name]['n']:,} "
                   f"pos_rate={splits_meta[name]['positive_rate']:.4f}")

    diag = {
        "n_original_samples": int(n_orig),
        "split_scheme": "three_axis_holdout_v5_12sets",
        "holdout_cell_types": holdout_cts,
        "val_holdout_cell_types": sorted(val_cell_set),
        "test_holdout_cell_types": sorted(test_cell_set),
        "main_cell_types": sorted(set(all_cell_types) - holdout_ct_set),
        "utr_holdout_frac": float(cfg.utr_holdout_frac),
        "val_utr_frac_of_holdout": float(cfg.val_utr_frac_of_holdout),
        "n_utr_total": len(unique_utrs),
        "n_utr_holdout_total": len(utr_h_set),
        "n_utr_val_holdout": len(val_utr_h_set),
        "n_utr_test_holdout": len(test_utr_h_set),
        "mirna_holdout_frac": float(cfg.mirna_holdout_frac),
        "n_mirna_total": len(unique_mirnas),
        "n_mirna_holdout": len(mirna_h_set),
        "n_unique_pairs_main_subspaces": int(n_unique_pairs),
        "n_unique_pairs_per_bucket": {
            "train": int(n_pair_train), "val": int(n_pair_val), "test": int(n_pair_test),
        },
        "both_label_group_survival": {
            "s_main":         [int(nk_main),  int(ng_main)],
            "s_o_o_tcell":    [int(nk_tcell), int(ng_tcell)],
            "s_o_o_vcell":    [int(nk_vcell), int(ng_vcell)],
            "s_vutr_o_omain": [int(nk_vutr),  int(ng_vutr)],
            "s_vutr_o_vcell": [int(nk_vuxc),  int(ng_vuxc)],
            "s_tutr_o_omain": [int(nk_tutr),  int(ng_tutr)],
            "s_tutr_o_tcell": [int(nk_uxc),   int(ng_uxc)],
            "s_o_mh_omain":   [int(nk_mh),    int(ng_mh)],
            "s_o_mh_tcell":   [int(nk_mhxc),  int(ng_mhxc)],
        },
        "utr_holdout_list": sorted(utr_h_set),
        "val_utr_holdout_list": sorted(val_utr_h_set),
        "test_utr_holdout_list": sorted(test_utr_h_set),
        "mirna_holdout_list": sorted(mirna_h_set),
        "splits": splits_meta,
    }

    return {
        "train": train_d,
        "val": val_d,
        "test_triple": test_triple_d,
        "val_cell_holdout": val_cell_holdout_d,
        "test_cell_holdout": test_cell_holdout_d,
        "test_pair_x_cell_holdout": test_pair_x_cell_holdout_d,
        "val_utr_holdout": val_utr_holdout_d,
        "val_utr_x_cell_holdout": val_utr_x_cell_holdout_d,
        "test_utr_holdout": test_utr_holdout_d,
        "test_utr_x_cell_holdout": test_utr_x_cell_holdout_d,
        "test_mirna_holdout": test_mirna_holdout_d,
        "test_mirna_x_cell_holdout": test_mirna_x_cell_holdout_d,
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
    for name in [
        "train", "val", "test_triple",
        "val_cell_holdout", "test_cell_holdout", "test_pair_x_cell_holdout",
        "val_utr_holdout", "val_utr_x_cell_holdout",
        "test_utr_holdout", "test_utr_x_cell_holdout",
        "test_mirna_holdout", "test_mirna_x_cell_holdout",
    ]:
        if name not in splits:
            continue
        sp = splits[name]
        for m, u in zip(sp["mirna_gid"].tolist(), sp["utr_gid"].tolist()):
            seen_pairs.add((m, u))

    n_pairs = len(seen_pairs)
    n_hit = 0
    n_fallback = 0
    # Also compute TargetScan hand-crafted features for each unique pair while we're
    # already touching every (mirna, utr) anchor — saves a second pass.
    sorted_pairs = sorted(seen_pairs)
    pair_to_row: dict[tuple[int, int], int] = {p: i for i, p in enumerate(sorted_pairs)}
    ts_feats = np.zeros((len(sorted_pairs), TARGETSCAN_FEATURE_DIM), dtype=np.float32)
    for i, (m, u) in enumerate(sorted_pairs):
        anc = find_seed_anchor(utr_seqs[u], seed_targets[m])
        if anc >= 0:
            n_hit += 1
        else:
            n_fallback += 1
        ts_feats[i] = compute_targetscan_features(utr_seqs[u], seed_targets[m])
    hit_rate = n_hit / max(n_pairs, 1)

    # persist pair index and feature table
    # pair_index encoded as parallel uint32 mirna_gid + utr_gid arrays for compact storage
    pair_mirna = np.array([p[0] for p in sorted_pairs], dtype=np.uint32)
    pair_utr = np.array([p[1] for p in sorted_pairs], dtype=np.uint32)
    np.savez(cache_dir / "targetscan_features.npz",
             pair_mirna=pair_mirna,
             pair_utr=pair_utr,
             features=ts_feats)
    ts_means = ts_feats.mean(axis=0) if len(ts_feats) else np.zeros(TARGETSCAN_FEATURE_DIM)
    ts_stds = ts_feats.std(axis=0) if len(ts_feats) else np.ones(TARGETSCAN_FEATURE_DIM)
    logger.log(f"[encode] targetscan features: n_pairs={len(sorted_pairs):,} "
               f"dim={TARGETSCAN_FEATURE_DIM} "
               f"mean=[{', '.join(f'{v:.3f}' for v in ts_means)}] "
               f"std=[{', '.join(f'{v:.3f}' for v in ts_stds)}]")

    report = {
        "seed_rule": seed_rule,
        "n_unique_pairs": n_pairs,
        "n_seed_anchor_found": n_hit,
        "n_seed_anchor_fallback": n_fallback,
        "seed_anchor_hit_rate": hit_rate,
        "seed_anchor_fallback_rate": 1.0 - hit_rate,
        "targetscan_feature_dim": TARGETSCAN_FEATURE_DIM,
        "targetscan_feature_means": ts_means.tolist(),
        "targetscan_feature_stds": ts_stds.tolist(),
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

    twelve_set_names = [
        "train", "val", "test_triple",
        "val_cell_holdout", "test_cell_holdout", "test_pair_x_cell_holdout",
        "val_utr_holdout", "val_utr_x_cell_holdout",
        "test_utr_holdout", "test_utr_x_cell_holdout",
        "test_mirna_holdout", "test_mirna_x_cell_holdout",
    ]
    for name in twelve_set_names:
        if name not in splits:
            continue
        d = splits[name]
        extra = {}
        if name == "train" and len(d["label"]) > 0:
            # ===== Per-row sample_weight = (group_balance) × (1 + 3·purity) =====
            # (1) GROUP BALANCE: per-(mirna, cell_id) groups.
            #     group_size = number of UTRs the model sees for this (mirna, cell) context;
            #     w_group = 1/sqrt(group_size) gives every group ≈ equal voice in the gradient.
            # (2) PAIR PURITY: per-(mirna, utr_gid) pairs.
            #     purity = std(labels across all cells for this pair); pure pairs (always 0 or
            #     always 1 across cells) have purity=0 — the model can solve those with a UTR
            #     shortcut. Mixed pairs (label flips across cells) have purity up to 0.5 — these
            #     are the rows that REQUIRE cell-awareness, so we upweight them.
            #     factor = (1 + 3·purity) ∈ [1, 2.5] : moderate boost so pure pairs still
            #     contribute (and class-balance via focal alpha is preserved).
            # Final: w = w_group × purity_factor, normalized so mean(w) = 1.
            mirna_list = d["mirna_gid"].tolist()
            utr_list = d["utr_gid"].tolist()
            cell_list = d["cell_id"].tolist()
            y_list = d["label"].tolist()

            # (1) group sizes
            group_size: dict[tuple[int, str], int] = {}
            for m, c in zip(mirna_list, cell_list):
                k = (int(m), str(c))
                group_size[k] = group_size.get(k, 0) + 1

            # (2) per-pair label std across cells (purity proxy)
            #     For each (m, u), aggregate the labels seen across (cell) contexts and
            #     take std. Cell-invariant pairs → 0; cell-dependent pairs → up to 0.5.
            pair_labels: dict[tuple[int, int], list[int]] = {}
            for m, u, y in zip(mirna_list, utr_list, y_list):
                pair_labels.setdefault((int(m), int(u)), []).append(int(y))
            pair_purity: dict[tuple[int, int], float] = {}
            for k_pu, ys in pair_labels.items():
                if len(ys) <= 1:
                    pair_purity[k_pu] = 0.0
                else:
                    pair_purity[k_pu] = float(np.std(ys, ddof=0))

            # combine
            w_group = np.array(
                [1.0 / np.sqrt(group_size[(int(m), str(c))])
                 for m, c in zip(mirna_list, cell_list)],
                dtype=np.float32,
            )
            w_purity = np.array(
                [1.0 + 3.0 * pair_purity[(int(m), int(u))]
                 for m, u in zip(mirna_list, utr_list)],
                dtype=np.float32,
            )
            sw = w_group * w_purity
            sw = sw / sw.mean()
            extra["sample_weight"] = sw

            # diagnostics
            n_mixed_pairs = sum(1 for v in pair_purity.values() if v > 0.0)
            n_total_pairs = len(pair_purity)
            mixed_frac = n_mixed_pairs / max(n_total_pairs, 1)
            logger.log(f"[cache] train sample_weight: n_groups={len(group_size):,} "
                       f"n_pairs={n_total_pairs:,} mixed_pairs={n_mixed_pairs:,} ({mixed_frac:.1%})")
            logger.log(f"[cache] train sample_weight stats: "
                       f"mean={sw.mean():.4f} std={sw.std():.4f} "
                       f"min={sw.min():.4f} max={sw.max():.4f} "
                       f"p50={np.percentile(sw, 50):.4f} p99={np.percentile(sw, 99):.4f}")
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
    # Persist holdout fractions for cache_exists() consistency check
    diag = splits["diagnostics"]
    (cache_dir / "holdout_config.json").write_text(json.dumps({
        "utr_holdout_frac": diag.get("utr_holdout_frac", 0.0),
        "mirna_holdout_frac": diag.get("mirna_holdout_frac", 0.0),
        "val_utr_frac_of_holdout": diag.get("val_utr_frac_of_holdout", 0.0),
        "holdout_cell_types": diag.get("holdout_cell_types", []),
        "val_holdout_cell_types": diag.get("val_holdout_cell_types", []),
        "split_scheme": diag.get("split_scheme", "unknown"),
    }), encoding="utf-8")

    # Pre-compute UTR-only trivial baselines on val/val_utr_holdout/val_cell_holdout
    # for normalized-lift composite early-stopping.
    train_d = splits["train"]
    train_pos_sum: dict[int, float] = {}
    train_count: dict[int, int] = {}
    for u, y in zip(train_d["utr_gid"].tolist(), train_d["label"].tolist()):
        train_count[u] = train_count.get(u, 0) + 1
        train_pos_sum[u] = train_pos_sum.get(u, 0.0) + float(y)
    train_utr_avg = {u: train_pos_sum[u] / train_count[u] for u in train_count}
    train_overall_pos = float(np.mean(train_d["label"])) if len(train_d["label"]) else 0.2

    def _utr_only_pr_auc(split_d):
        if len(split_d["label"]) == 0:
            return 0.0
        preds = np.array([train_utr_avg.get(int(u), train_overall_pos)
                          for u in split_d["utr_gid"].tolist()])
        # Compute average precision (PR-AUC) via numpy, no sklearn dependency
        y_true = np.asarray(split_d["label"], dtype=np.float64)
        order = np.argsort(-preds)
        yt = y_true[order]
        tp = np.cumsum(yt)
        fp = np.cumsum(1.0 - yt)
        p_tot = float(yt.sum())
        if p_tot == 0:
            return 0.0
        precision = tp / np.maximum(tp + fp, 1.0)
        recall = tp / p_tot
        r_prev = np.concatenate([[0.0], recall[:-1]])
        return float(((recall - r_prev) * precision).sum())

    baselines = {}
    for name in ["val", "val_utr_holdout", "val_cell_holdout", "val_utr_x_cell_holdout"]:
        if name in splits and len(splits[name]["label"]) > 0:
            baselines[name] = _utr_only_pr_auc(splits[name])
    (cache_dir / "early_stop_baselines.json").write_text(
        json.dumps(baselines, indent=2), encoding="utf-8")
    logger.log(f"[cache] UTR-only baselines for early stopping: {baselines}")
    logger.log(f"[cache] wrote cache to {cache_dir}")


def cache_exists(cache_dir: Path, expected_seed_rule: str | None = None,
                 expected_utr_holdout_frac: float | None = None,
                 expected_mirna_holdout_frac: float | None = None,
                 expected_holdout_cell_types: list[str] | None = None,
                 expected_val_holdout_cell_types: list[str] | None = None) -> bool:
    """Return True only if the cache is complete AND every expected_* matches."""
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
            return False
        if rule_file.read_text(encoding="utf-8").strip() != expected_seed_rule:
            return False
    # check holdout config (new for 9-set design)
    hc_path = cache_dir / "holdout_config.json"
    if expected_utr_holdout_frac is not None or expected_mirna_holdout_frac is not None \
            or expected_holdout_cell_types is not None \
            or expected_val_holdout_cell_types is not None:
        if not hc_path.exists():
            return False  # old 5-set cache; force rebuild
        try:
            hc = json.loads(hc_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        if expected_utr_holdout_frac is not None and \
                abs(float(hc.get("utr_holdout_frac", -1)) - expected_utr_holdout_frac) > 1e-6:
            return False
        if expected_mirna_holdout_frac is not None and \
                abs(float(hc.get("mirna_holdout_frac", -1)) - expected_mirna_holdout_frac) > 1e-6:
            return False
        if expected_holdout_cell_types is not None and \
                sorted(hc.get("holdout_cell_types", [])) != sorted(expected_holdout_cell_types):
            return False
        if expected_val_holdout_cell_types is not None and \
                sorted(hc.get("val_holdout_cell_types", [])) != sorted(expected_val_holdout_cell_types):
            return False
    return True


def load_cache(cache_dir: Path, logger: Logger) -> dict[str, Any]:
    mirna_seqs = json.loads(
        (cache_dir / "global_maps" / "mirna_seqs.json").read_text(encoding="utf-8"))
    with open(cache_dir / "global_maps" / "utr_seqs.pkl", "rb") as fh:
        utr_seqs = pickle.load(fh)
    cell_type_to_idx = json.loads(
        (cache_dir / "global_maps" / "cell_type_to_idx.json").read_text(encoding="utf-8"))
    mirna_codes_global = np.load(cache_dir / "mirna_codes_global.npy")
    seed_targets = json.loads(
        (cache_dir / "seed_targets.json").read_text(encoding="utf-8"))
    logger.log(f"[cache] loaded: n_mirna={len(mirna_seqs)} n_utr={len(utr_seqs)} "
               f"n_cell_types={len(cell_type_to_idx)}")

    # TargetScan hand-crafted features: dict[(mirna_gid, utr_gid) -> 5-vec]
    ts_feat_dict: dict[tuple[int, int], np.ndarray] = {}
    ts_feat_means = np.zeros(TARGETSCAN_FEATURE_DIM, dtype=np.float32)
    ts_feat_stds = np.ones(TARGETSCAN_FEATURE_DIM, dtype=np.float32)
    ts_path = cache_dir / "targetscan_features.npz"
    if ts_path.exists():
        ts_z = np.load(ts_path, allow_pickle=False)
        pm = ts_z["pair_mirna"].astype(np.int64)
        pu = ts_z["pair_utr"].astype(np.int64)
        pf = ts_z["features"].astype(np.float32)
        for i in range(len(pm)):
            ts_feat_dict[(int(pm[i]), int(pu[i]))] = pf[i]
        # Read normalization stats from the seed_anchor_report if available; otherwise
        # recompute them right here.
        try:
            rep = json.loads((cache_dir / "seed_anchor_report.json").read_text(encoding="utf-8"))
            if "targetscan_feature_means" in rep and "targetscan_feature_stds" in rep:
                ts_feat_means = np.array(rep["targetscan_feature_means"], dtype=np.float32)
                ts_feat_stds = np.array(rep["targetscan_feature_stds"], dtype=np.float32)
        except Exception:
            pass
        if not np.all(np.isfinite(ts_feat_means)):
            ts_feat_means = pf.mean(axis=0)
        if not np.all(np.isfinite(ts_feat_stds)):
            ts_feat_stds = pf.std(axis=0)
        # avoid div-by-zero
        ts_feat_stds = np.where(ts_feat_stds < 1e-6, 1.0, ts_feat_stds).astype(np.float32)
        logger.log(f"[cache] loaded targetscan features: n_pairs={len(ts_feat_dict):,}")
    splits: dict[str, dict[str, np.ndarray]] = {}
    twelve_set_names = [
        "train", "val", "test_triple",
        "val_cell_holdout", "test_cell_holdout", "test_pair_x_cell_holdout",
        "val_utr_holdout", "val_utr_x_cell_holdout",
        "test_utr_holdout", "test_utr_x_cell_holdout",
        "test_mirna_holdout", "test_mirna_x_cell_holdout",
    ]
    for name in twelve_set_names:
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
        "mirna_codes_global": mirna_codes_global,
        "seed_targets": seed_targets,
        "ts_feat_dict": ts_feat_dict,
        "ts_feat_means": ts_feat_means,
        "ts_feat_stds": ts_feat_stds,
        "splits": splits,
    }


# ---------------------------------------------------------------------------
# V4 diagnostic: mixed-pair-only train filter (mirrors train_main_model_mixed_only.py).
# Kept inlined here so V4 is self-contained — no cross-script imports.
# ---------------------------------------------------------------------------

def filter_train_to_mixed_only(cache: dict, logger: Logger) -> dict | None:
    """In-place filter cache['splits']['train'] to mixed pairs only.

    A pair (mirna_gid, utr_gid) is 'mixed' iff its labels across cells are not
    all the same value. Pure pairs (label invariant across cells) are dropped.
    Recomputes sample_weight and UTR-only PR-AUC baselines on the filtered set.
    Returns recomputed baselines dict, or None if train was empty.
    """
    train = cache["splits"]["train"]
    if len(train["label"]) == 0:
        logger.log("[mixed-only] train is empty; skipping filter")
        return None

    mirna_arr = train["mirna_gid"].tolist()
    utr_arr   = train["utr_gid"].tolist()
    cell_arr  = train["cell_id"].tolist()
    y_arr     = train["label"].tolist()

    pair_label_set: dict[tuple[int, int], set] = {}
    for m, u, y in zip(mirna_arr, utr_arr, y_arr):
        pair_label_set.setdefault((int(m), int(u)), set()).add(int(y))
    mixed_pairs = {k for k, ys in pair_label_set.items() if len(ys) > 1}
    n_pairs_total = len(pair_label_set)
    n_pairs_mixed = len(mixed_pairs)

    keep_mask = np.array(
        [(int(m), int(u)) in mixed_pairs for m, u in zip(mirna_arr, utr_arr)],
        dtype=bool,
    )
    n_before = len(train["label"])
    n_after = int(keep_mask.sum())
    if n_after == 0:
        logger.log("[mixed-only] no mixed pairs in train; aborting filter")
        return None

    for k, v in list(train.items()):
        if isinstance(v, np.ndarray) and v.shape[0] == n_before:
            train[k] = v[keep_mask]

    logger.log(
        f"[mixed-only] pairs: {n_pairs_mixed:,}/{n_pairs_total:,} mixed "
        f"({n_pairs_mixed / max(n_pairs_total, 1):.1%})  |  "
        f"rows: {n_before:,} -> {n_after:,} ({n_after / max(n_before, 1):.1%} retained)"
    )

    new_mirna = train["mirna_gid"].tolist()
    new_utr   = train["utr_gid"].tolist()
    new_cell  = train["cell_id"].tolist()
    new_y     = train["label"].tolist()

    group_size: dict[tuple[int, str], int] = {}
    for m, c in zip(new_mirna, new_cell):
        k = (int(m), str(c))
        group_size[k] = group_size.get(k, 0) + 1

    pair_labels_new: dict[tuple[int, int], list[int]] = {}
    for m, u, y in zip(new_mirna, new_utr, new_y):
        pair_labels_new.setdefault((int(m), int(u)), []).append(int(y))
    pair_purity_new = {
        k: (float(np.std(ys, ddof=0)) if len(ys) > 1 else 0.0)
        for k, ys in pair_labels_new.items()
    }

    w_group = np.array(
        [1.0 / np.sqrt(group_size[(int(m), str(c))])
         for m, c in zip(new_mirna, new_cell)],
        dtype=np.float32,
    )
    w_purity = np.array(
        [1.0 + 3.0 * pair_purity_new[(int(m), int(u))]
         for m, u in zip(new_mirna, new_utr)],
        dtype=np.float32,
    )
    sw = w_group * w_purity
    if sw.mean() > 0:
        sw = sw / sw.mean()
    train["sample_weight"] = sw
    logger.log(
        f"[mixed-only] new train sample_weight: n_groups={len(group_size):,}  "
        f"mean={sw.mean():.4f} std={sw.std():.4f} "
        f"min={sw.min():.4f} max={sw.max():.4f}"
    )

    train_pos_sum: dict[int, float] = {}
    train_count: dict[int, int] = {}
    for u, y in zip(new_utr, new_y):
        train_count[u] = train_count.get(u, 0) + 1
        train_pos_sum[u] = train_pos_sum.get(u, 0.0) + float(y)
    train_utr_avg = {u: train_pos_sum[u] / train_count[u] for u in train_count}
    train_overall_pos = float(np.mean(new_y)) if new_y else 0.2

    def _utr_only_pr_auc(split_d):
        if len(split_d["label"]) == 0:
            return 0.0
        preds = np.array([train_utr_avg.get(int(u), train_overall_pos)
                          for u in split_d["utr_gid"].tolist()])
        y_true = np.asarray(split_d["label"], dtype=np.float64)
        order = np.argsort(-preds)
        yt = y_true[order]
        tp = np.cumsum(yt)
        fp = np.cumsum(1.0 - yt)
        p_tot = float(yt.sum())
        if p_tot == 0:
            return 0.0
        precision = tp / np.maximum(tp + fp, 1.0)
        recall = tp / p_tot
        r_prev = np.concatenate([[0.0], recall[:-1]])
        return float(((recall - r_prev) * precision).sum())

    new_baselines: dict[str, float] = {}
    for name in ["val", "val_utr_holdout", "val_cell_holdout", "val_utr_x_cell_holdout"]:
        if name in cache["splits"] and len(cache["splits"][name]["label"]) > 0:
            new_baselines[name] = _utr_only_pr_auc(cache["splits"][name])
    logger.log(f"[mixed-only] new UTR-only baselines (from filtered train): {new_baselines}")
    return new_baselines


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
                 utr_seqs: list[str],
                 ts_feat_dict: dict | None = None,
                 ts_feat_means: np.ndarray | None = None,
                 ts_feat_stds: np.ndarray | None = None,
                 utr_window_len: int = UTR_WINDOW_LEN,
                 build_seed_mask: bool = True,
                 seed_mask_downsample: int = UTR_DOWNSAMPLE_V4):
        # ----- V4 NOTE -----
        # `utr_window_len` is now passed in (default 4000) so we can change the
        # window per run without touching the module-level constant.
        # `build_seed_mask` toggles on the per-position seed-match label that
        # the V4 auxiliary head supervises against. When False, mask is zeros
        # (used for eval/inference, where aux loss is not computed).
        # `seed_mask_downsample` matches the ConvNeXt encoder's total stride.
        self.utr_window_len = int(utr_window_len)
        self.build_seed_mask = bool(build_seed_mask)
        self.seed_mask_ds = int(seed_mask_downsample)
        self.mask_len = self.utr_window_len // self.seed_mask_ds
        self.mirna_gid = rows["mirna_gid"].astype(np.int64)
        self.utr_gid = rows["utr_gid"].astype(np.int64)
        self.cell_idx = rows["cell_idx"].astype(np.int64)
        self.label = rows["label"].astype(np.float32)
        # sample_weight is only present in train rows (eval/val default to 1.0)
        if "sample_weight" in rows:
            self.sample_weight = rows["sample_weight"].astype(np.float32)
        else:
            self.sample_weight = np.ones(len(self.label), dtype=np.float32)
        self.mirna_codes_global = mirna_codes_global
        self.seed_targets = seed_targets
        self.utr_seqs = utr_seqs
        # ===== TargetScan-style hand-crafted features =====
        # Pre-resolve per-row feature vector (z-scored) for speed. Falls back to
        # zeros if the pair was not in the precomputed table (shouldn't happen
        # in normal use since the table covers every pair seen in any split).
        self._ts_dim = TARGETSCAN_FEATURE_DIM
        self.has_ts_feats = (ts_feat_dict is not None and len(ts_feat_dict) > 0)
        if self.has_ts_feats:
            means = ts_feat_means if ts_feat_means is not None else \
                np.zeros(self._ts_dim, dtype=np.float32)
            stds = ts_feat_stds if ts_feat_stds is not None else \
                np.ones(self._ts_dim, dtype=np.float32)
            stds = np.where(stds < 1e-6, 1.0, stds).astype(np.float32)
            zero_vec = (np.zeros(self._ts_dim, dtype=np.float32) - means) / stds
            ts_rows = np.empty((len(self.label), self._ts_dim), dtype=np.float32)
            for i in range(len(self.label)):
                k = (int(self.mirna_gid[i]), int(self.utr_gid[i]))
                v = ts_feat_dict.get(k)
                if v is None:
                    ts_rows[i] = zero_vec
                else:
                    ts_rows[i] = (v.astype(np.float32) - means) / stds
            self.ts_feats = ts_rows
        else:
            self.ts_feats = np.zeros((len(self.label), self._ts_dim), dtype=np.float32)

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
        W = self.utr_window_len
        cropped, _, _ = crop_utr_seed_anchored(utr_seq, seed_target, W)
        utr_codes = encode_one_hot(cropped, W)
        utr_oh = codes_to_one_hot(utr_codes)
        v_len = min(len(cropped), W)
        # V4: per-position seed-match mask, downsampled to encoder output resolution
        seed_mask = self._compute_seed_mask(cropped, seed_target)
        return (
            mirna_oh,
            utr_oh,
            np.int32(v_len),
            np.int64(self.cell_idx[idx]),
            self.ts_feats[idx].astype(np.float32),
            seed_mask,
            np.float32(self.sample_weight[idx]),
            np.float32(self.label[idx]),
        )

    def _compute_seed_mask(self, cropped_utr: str, seed_target) -> np.ndarray:
        """V4: build a downsampled binary mask marking which encoder-output
        positions contain at least one seed-target match in their receptive
        field. Returns float32 array of shape (mask_len,).

        If build_seed_mask is False, returns zeros (eval/inference path)."""
        L_out = self.mask_len
        if not self.build_seed_mask:
            return np.zeros(L_out, dtype=np.float32)
        ds = self.seed_mask_ds
        # find positions of all matches in the cropped sequence
        positions = find_all_seed_matches(cropped_utr, seed_target)
        if not positions:
            return np.zeros(L_out, dtype=np.float32)
        mask = np.zeros(L_out, dtype=np.float32)
        for p in positions:
            bucket = p // ds
            if 0 <= bucket < L_out:
                mask[bucket] = 1.0
        return mask


def collate_batch(batch):
    import torch
    mirna_x = torch.from_numpy(np.stack([x[0] for x in batch]))
    utr_x = torch.from_numpy(np.stack([x[1] for x in batch]))
    utr_vlen = torch.from_numpy(np.asarray([x[2] for x in batch], dtype=np.int32))
    cell_x = torch.from_numpy(np.asarray([x[3] for x in batch], dtype=np.int64))
    ts_feats = torch.from_numpy(np.stack([x[4] for x in batch]).astype(np.float32))
    seed_mask = torch.from_numpy(np.stack([x[5] for x in batch]).astype(np.float32))  # V4
    weight = torch.from_numpy(np.asarray([x[6] for x in batch], dtype=np.float32))
    y = torch.from_numpy(np.asarray([x[7] for x in batch], dtype=np.float32))
    return mirna_x, utr_x, utr_vlen, cell_x, ts_feats, seed_mask, weight, y


# ---------------------------------------------------------------------------
# Model: miRNA + UTR (seed-anchored, attention-pooled) + cell embedding
# ---------------------------------------------------------------------------

def weighted_focal_loss(logits, labels, sample_weight, gamma: float = 2.0,
                         pos_weight_val: float = 1.0):
    """Per-sample weighted focal loss with class-balance alpha.

    loss_i = alpha_t * (1 - p_t)^gamma * BCE(logit_i, y_i) * sample_weight_i

    - alpha_t balances positive vs negative class (alpha=pos_weight/(1+pos_weight))
      so e.g. pos_weight=4 gives alpha=0.8 for positives.
    - (1 - p_t)^gamma is the focal factor: down-weights easy examples (high p_t),
      up-weights hard examples (low p_t). gamma=2 is the standard default.
    - sample_weight is the per-(mirna, cell_id) group-balance weight, computed
      at cache build time, that makes every (mirna, cell) context equally
      heard regardless of how many UTRs it has.
    """
    import torch
    import torch.nn.functional as F
    p = torch.sigmoid(logits)
    p_t = labels * p + (1.0 - labels) * (1.0 - p)
    alpha_pos = pos_weight_val / (1.0 + pos_weight_val)
    alpha_t = labels * alpha_pos + (1.0 - labels) * (1.0 - alpha_pos)
    focal_factor = (1.0 - p_t) ** gamma
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    return (alpha_t * focal_factor * bce * sample_weight).mean()


def aux_seed_mask_loss(aux_logits, seed_mask, utr_vlen, downsample: int = 16):
    """V4 auxiliary loss: per-position BCE on the seed-match mask.

    aux_logits  : [B, L_pool]  raw logits from aux head
    seed_mask   : [B, L_pool]  ground-truth binary mask (1 = seed match here)
    utr_vlen    : [B]          original UTR window valid length (in nt)
    downsample  : encoder total stride (used to mask invalid pool positions)

    Positions beyond the valid (un-padded) UTR length are excluded from the
    loss. Class imbalance (most positions are 0) is handled with a pos_weight
    estimated from the batch.
    """
    import torch
    import torch.nn.functional as F
    B, L = aux_logits.shape
    valid_pooled = (utr_vlen.float() / float(downsample)).ceil().long()
    valid_pooled = torch.clamp(valid_pooled, min=1, max=L)
    pos_idx = torch.arange(L, device=aux_logits.device).unsqueeze(0)
    valid_mask = (pos_idx < valid_pooled.unsqueeze(1)).float()  # [B, L]

    # auto pos_weight per batch (rare positives in seed mask)
    n_pos = (seed_mask * valid_mask).sum().clamp(min=1.0)
    n_neg = ((1.0 - seed_mask) * valid_mask).sum().clamp(min=1.0)
    pw = (n_neg / n_pos).clamp(max=50.0)  # cap to avoid extreme weight

    per_pos = F.binary_cross_entropy_with_logits(
        aux_logits, seed_mask, reduction='none',
        pos_weight=pw,
    )
    per_pos = per_pos * valid_mask
    total = valid_mask.sum().clamp(min=1.0)
    return per_pos.sum() / total


def make_model(n_cell_types: int, model_kind: str = "full",
               cell_embed_dim: int = 32, use_ts_features: bool = True,
               ts_feature_dim: int = TARGETSCAN_FEATURE_DIM,
               # ----- V4 additions -----
               use_convnext: bool = True,
               use_aux_seed_head: bool = True,
               convnext_dims=(64, 128, 192, 256),
               convnext_depths=(3, 3, 6, 2)):
    """V4 model factory. model_kind options unchanged from V3:
      'full' | 'no_cell' | 'seq_only' | 'no_attn' | 'cell_only' | 'film'

    V4 additions:
      use_convnext        : replace 3-layer vanilla Conv UTR branch with a
                            4-stage ConvNeXt-1D encoder (~1.5M params).
      use_aux_seed_head   : add a per-position seed-match prediction head on
                            the UTR encoder output (pre-FiLM) for auxiliary
                            supervision. Forward then returns
                                (main_logits, aux_seed_logits)
                            instead of just main_logits.
      convnext_dims       : channels per stage (V4 default: 64,128,192,256)
      convnext_depths     : ConvNeXt blocks per stage (V4 default: 3,3,6,2)
    """
    import torch
    from torch import nn
    import torch.nn.functional as F

    USE_SEQ = model_kind in {"full", "no_cell", "seq_only", "no_attn", "film"}
    USE_CELL = model_kind in {"full", "cell_only", "no_attn", "film"}
    USE_ATTN = model_kind in {"full", "no_cell", "seq_only", "film"}
    USE_FILM = (model_kind == "film")
    CELL_CONCAT_AT_HEAD = USE_CELL and not USE_FILM
    USE_TS = bool(use_ts_features) and ts_feature_dim > 0
    USE_CNXT = bool(use_convnext) and USE_SEQ
    USE_AUX = bool(use_aux_seed_head) and USE_SEQ
    UTR_OUT_DIM = 128   # output channels of the UTR encoder (project ConvNeXt to this)
    UTR_TOTAL_STRIDE = 16   # 4kb -> 250 (stem stride 2 + 3 downsamples of 2 between 4 stages)

    # -------------------------- ConvNeXt-1D block --------------------------
    class ConvNeXt1DBlock(nn.Module):
        """Depthwise-conv + LayerNorm + inverted-bottleneck MLP, with residual.

        Faithful 1D translation of the ConvNeXt block (Liu et al. 2022;
        REPRESS Fig g uses the same shape and reports +17% over dilated CNN
        and +10% over Mamba state-space at the same parameter count)."""
        def __init__(self, dim: int, kernel: int = 7, mlp_ratio: int = 4,
                     dropout: float = 0.0):
            super().__init__()
            pad = kernel // 2
            self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel, padding=pad,
                                    groups=dim)
            self.norm = nn.LayerNorm(dim)
            self.pwconv1 = nn.Linear(dim, mlp_ratio * dim)
            self.act = nn.GELU()
            self.pwconv2 = nn.Linear(mlp_ratio * dim, dim)
            self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        def forward(self, x):
            # x: [B, C, L]
            residual = x
            x = self.dwconv(x)                          # [B, C, L]
            x = x.permute(0, 2, 1)                       # [B, L, C]
            x = self.norm(x)
            x = self.pwconv1(x)
            x = self.act(x)
            x = self.pwconv2(x)
            x = self.drop(x)
            x = x.permute(0, 2, 1)                       # [B, C, L]
            return residual + x

    # ---- a stage of N ConvNeXt blocks at fixed channel width ----
    def _stage(dim: int, n_blocks: int, kernel: int = 7) -> nn.Sequential:
        return nn.Sequential(*[ConvNeXt1DBlock(dim, kernel=kernel)
                               for _ in range(n_blocks)])

    class CellAwareMTI(nn.Module):
        def __init__(self):
            super().__init__()
            head_in = 0

            if USE_SEQ:
                # ---- miRNA encoder (unchanged from V3: small, 22 nt) ----
                self.mirna_branch = nn.Sequential(
                    nn.Conv1d(4, 32, kernel_size=7, padding=3),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.Conv1d(32, 64, kernel_size=5, padding=2),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                )
                self.mirna_pool = nn.AdaptiveAvgPool1d(1)

                # ---- UTR encoder ----
                if USE_CNXT:
                    d0, d1, d2, d3 = convnext_dims
                    n0, n1, n2, n3 = convnext_depths
                    # stem: 4 -> d0, stride 2 (4000 -> 2000)
                    self.utr_stem = nn.Sequential(
                        nn.Conv1d(4, d0, kernel_size=7, stride=2, padding=3),
                        # LayerNorm wants channel-last, do it in a small adapter:
                    )
                    self.utr_stem_norm = nn.LayerNorm(d0)
                    # 4 stages with downsample between (each downsample halves length)
                    self.utr_stage1 = _stage(d0, n0)                            # 2000
                    self.utr_down1  = nn.Conv1d(d0, d1, kernel_size=2, stride=2)
                    self.utr_stage2 = _stage(d1, n1)                            # 1000
                    self.utr_down2  = nn.Conv1d(d1, d2, kernel_size=2, stride=2)
                    self.utr_stage3 = _stage(d2, n2)                            #  500
                    self.utr_down3  = nn.Conv1d(d2, d3, kernel_size=2, stride=2)
                    self.utr_stage4 = _stage(d3, n3)                            #  250
                    # project to UTR_OUT_DIM = 128 so downstream FiLM / attention shapes match V3
                    self.utr_project = nn.Conv1d(d3, UTR_OUT_DIM, kernel_size=1)
                    self.utr_downsample = UTR_TOTAL_STRIDE  # for attention mask building
                else:
                    # ---- fallback: V3 vanilla 3-layer CNN ----
                    self.utr_branch = nn.Sequential(
                        nn.Conv1d(4, 32, kernel_size=9, padding=4),
                        nn.BatchNorm1d(32), nn.ReLU(),
                        nn.MaxPool1d(4),
                        nn.Conv1d(32, 64, kernel_size=7, padding=3),
                        nn.BatchNorm1d(64), nn.ReLU(),
                        nn.MaxPool1d(4),
                        nn.Conv1d(64, UTR_OUT_DIM, kernel_size=5, padding=2),
                        nn.BatchNorm1d(UTR_OUT_DIM), nn.ReLU(),
                    )
                    self.utr_downsample = 16

                if USE_ATTN:
                    self.mirna_to_q = nn.Linear(64, UTR_OUT_DIM)
                head_in += 64 + UTR_OUT_DIM

                # ---- V4 auxiliary head: per-position seed-match logits ----
                # 1x1 conv on UTR encoder output (pre-FiLM) -> [B, 1, L_u]
                if USE_AUX:
                    self.aux_seed_head = nn.Conv1d(UTR_OUT_DIM, 1, kernel_size=1)

            if USE_CELL:
                self.cell_embedding = nn.Embedding(n_cell_types, cell_embed_dim)
                if CELL_CONCAT_AT_HEAD:
                    head_in += cell_embed_dim

            if USE_FILM:
                self.film_gen = nn.Sequential(
                    nn.Linear(cell_embed_dim, UTR_OUT_DIM),
                    nn.ReLU(),
                    nn.Linear(UTR_OUT_DIM, 2 * UTR_OUT_DIM),
                )

            if USE_TS:
                ts_hidden = 16
                self.ts_branch = nn.Sequential(
                    nn.Linear(ts_feature_dim, ts_hidden),
                    nn.ReLU(),
                    nn.Linear(ts_hidden, ts_hidden),
                    nn.ReLU(),
                )
                head_in += ts_hidden

            assert head_in > 0, "no active branches"
            # V4: slightly wider head (256 hidden) to match larger encoder capacity
            head_hidden = 256 if USE_CNXT else 128
            self.head = nn.Sequential(
                nn.Linear(head_in, head_hidden),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(head_hidden, 1),
            )

            self.use_seq = USE_SEQ
            self.use_cell = USE_CELL
            self.use_attn = USE_ATTN
            self.use_film = USE_FILM
            self.use_ts = USE_TS
            self.use_convnext = USE_CNXT
            self.use_aux = USE_AUX
            self.cell_concat_at_head = CELL_CONCAT_AT_HEAD

        def _encode_utr(self, utr_x):
            """Return UTR feature map [B, UTR_OUT_DIM, L_pooled]."""
            if self.use_convnext:
                x = self.utr_stem(utr_x)                     # [B, d0, L/2]
                # LayerNorm in channel-last
                x = x.permute(0, 2, 1)
                x = self.utr_stem_norm(x)
                x = x.permute(0, 2, 1)
                x = self.utr_stage1(x)
                x = self.utr_down1(x)
                x = self.utr_stage2(x)
                x = self.utr_down2(x)
                x = self.utr_stage3(x)
                x = self.utr_down3(x)
                x = self.utr_stage4(x)
                x = self.utr_project(x)                       # [B, UTR_OUT_DIM, L_pooled]
                return x
            else:
                return self.utr_branch(utr_x)

        def _build_utr_attn_mask(self, utr_vlen, u_seq_len: int):
            valid_pooled = (utr_vlen.float() / self.utr_downsample).ceil().long()
            valid_pooled = torch.clamp(valid_pooled, min=1, max=u_seq_len)
            arange = torch.arange(u_seq_len, device=utr_vlen.device).unsqueeze(0)
            return arange < valid_pooled.unsqueeze(1)

        def forward(self, mirna_x, utr_x, utr_vlen, cell_x, ts_feats=None):
            """V4 forward.

            Returns:
                if use_aux:  (main_logits [B], aux_seed_logits [B, L_pooled])
                else:        main_logits [B]
            """
            parts = []
            cell_emb = None
            if self.use_cell:
                cell_safe = torch.clamp(cell_x, min=0)
                cell_emb = self.cell_embedding(cell_safe)

            aux_logits = None

            if self.use_seq:
                m_seq = self.mirna_branch(mirna_x)
                m = self.mirna_pool(m_seq).squeeze(-1)            # [B, 64]
                u_seq = self._encode_utr(utr_x)                    # [B, 128, L_pool]

                # === V4 aux head: predict seed match positions BEFORE FiLM ===
                # (predicting before cell modulation keeps the aux task purely
                # sequence-level — what the encoder must learn first.)
                if self.use_aux:
                    aux_logits = self.aux_seed_head(u_seq).squeeze(1)  # [B, L_pool]

                # === Optional FiLM modulation ===
                if self.use_film and cell_emb is not None:
                    film_params = self.film_gen(cell_emb)              # [B, 2C]
                    gamma, beta = film_params.chunk(2, dim=-1)
                    u_seq = u_seq * (1.0 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)

                # === Pool UTR ===
                if self.use_attn:
                    q = self.mirna_to_q(m).unsqueeze(1)                # [B, 1, C]
                    attn = torch.bmm(q, u_seq).squeeze(1)              # [B, L_pool]
                    mask = self._build_utr_attn_mask(utr_vlen, u_seq.shape[-1])
                    attn = attn.masked_fill(~mask, float("-inf"))
                    all_masked = (~mask).all(dim=-1)
                    if all_masked.any():
                        attn[all_masked, 0] = 0.0
                    w = F.softmax(attn, dim=-1)
                    u = torch.bmm(u_seq, w.unsqueeze(-1)).squeeze(-1)  # [B, C]
                else:
                    mask = self._build_utr_attn_mask(utr_vlen, u_seq.shape[-1])
                    m_f = mask.float().unsqueeze(1)
                    u = (u_seq * m_f).sum(dim=-1) / m_f.sum(dim=-1).clamp(min=1.0)

                parts.append(m)
                parts.append(u)

            if self.cell_concat_at_head and cell_emb is not None:
                parts.append(cell_emb)
            if self.use_ts and ts_feats is not None:
                parts.append(self.ts_branch(ts_feats))

            z = torch.cat(parts, dim=1)
            main_logits = self.head(z).squeeze(1)

            if self.use_aux:
                return main_logits, aux_logits
            return main_logits

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
        # V4: collate emits 8-tuple (incl. seed_mask). We discard seed_mask here
        # because the aux loss is only used at training time.
        for mirna_x, utr_x, utr_vlen, cell_x, ts_feats, _seed_mask, _weight, y in loader:
            mirna_x = mirna_x.to(device, non_blocking=True)
            utr_x = utr_x.to(device, non_blocking=True)
            utr_vlen = utr_vlen.to(device, non_blocking=True)
            cell_x = cell_x.to(device, non_blocking=True)
            ts_feats = ts_feats.to(device, non_blocking=True)
            y_dev = y.to(device, non_blocking=True)
            out = model(mirna_x, utr_x, utr_vlen, cell_x, ts_feats)
            # V4: model may return (main_logits, aux_logits). For eval we only
            # care about main_logits.
            logits = out[0] if isinstance(out, tuple) else out
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


def train_one_epoch(model, loader, loss_fn, opt, device, logger, log_every: int, epoch: int,
                    aux_weight: float = 0.0, utr_downsample: int = 16):
    """V4 train loop. loss_fn returns the MAIN repression loss as a tensor.
    The aux seed-mask loss is computed here if the model returns (main, aux).
    Total loss = main_loss + aux_weight * aux_loss.
    """
    model.train()
    total_main = 0.0
    total_aux = 0.0
    n_batches = 0
    total_batches = len(loader)
    t0 = time.time()
    for bidx, (mirna_x, utr_x, utr_vlen, cell_x, ts_feats, seed_mask, weight, y) in enumerate(loader, start=1):
        mirna_x = mirna_x.to(device, non_blocking=True)
        utr_x = utr_x.to(device, non_blocking=True)
        utr_vlen = utr_vlen.to(device, non_blocking=True)
        cell_x = cell_x.to(device, non_blocking=True)
        ts_feats = ts_feats.to(device, non_blocking=True)
        seed_mask = seed_mask.to(device, non_blocking=True)
        weight = weight.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        opt.zero_grad()
        out = model(mirna_x, utr_x, utr_vlen, cell_x, ts_feats)
        if isinstance(out, tuple):
            logits, aux_logits = out
        else:
            logits, aux_logits = out, None

        main_loss = loss_fn(logits, y, weight)
        loss = main_loss
        aux_val = 0.0
        if aux_logits is not None and aux_weight > 0:
            a_loss = aux_seed_mask_loss(aux_logits, seed_mask, utr_vlen,
                                        downsample=utr_downsample)
            loss = loss + aux_weight * a_loss
            aux_val = float(a_loss.detach().cpu().item())

        loss.backward()
        opt.step()
        total_main += float(main_loss.detach().cpu().item())
        total_aux += aux_val
        n_batches += 1
        if log_every > 0 and (bidx % log_every == 0 or bidx == total_batches):
            elapsed = time.time() - t0
            eta = elapsed / max(bidx, 1) * max(total_batches - bidx, 0)
            running_main = total_main / n_batches
            running_aux = total_aux / n_batches
            extra = f" aux_loss={running_aux:.4f}" if aux_weight > 0 else ""
            logger.log(
                f"[train] epoch={epoch} batch={bidx:,}/{total_batches:,} "
                f"running_main={running_main:.4f}{extra} elapsed={elapsed:.1f}s eta={eta:.1f}s"
            )
    return total_main / max(n_batches, 1)


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
    utr_holdout_frac: float = 0.20
    mirna_holdout_frac: float = 0.20
    val_holdout_cell_types: list[str] | None = None
    loss_kind: str = "focal_balanced"   # 'bce' (legacy) or 'focal_balanced' (focal + per-group weights)
    focal_gamma: float = 2.0
    use_ts_features: bool = True        # concat 5-d TargetScan-style hand-crafted features at head
    # ----- V4 additions -----
    utr_window_len: int = 4000          # V4 default; V3 used 2000
    use_aux_seed_head: bool = True      # V4 default; per-position seed-match aux supervision
    aux_seed_weight: float = 0.3        # weight on auxiliary seed-mask BCE loss
    use_convnext: bool = True           # V4 default; if False, falls back to V3 vanilla CNN
    convnext_dims: tuple = (64, 128, 192, 256)  # per-stage channels
    convnext_depths: tuple = (3, 3, 6, 2)       # blocks per stage
    train_mixed_only: bool = False      # V4 diagnostic: filter train to mixed pairs only (default OFF)


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--run_tag", default="main_v1")
    p.add_argument("--model", choices=["full", "no_cell", "seq_only", "no_attn", "cell_only", "film"], default="full")
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
    p.add_argument("--utr_holdout_frac", type=float, default=0.20,
                   help="fraction of UTRs to hold out entirely (stratified by per-UTR pos_rate)")
    p.add_argument("--mirna_holdout_frac", type=float, default=0.20,
                   help="fraction of miRNAs to hold out entirely (stratified by per-miRNA pos_rate)")
    p.add_argument("--val_holdout_cell_types", default="",
                   help="comma-separated cell_types reserved for val_cell_holdout (early stopping); "
                        "must be a subset of --holdout_cell_types; default = auto-pick 1")
    p.add_argument("--loss_kind", default="focal_balanced",
                   choices=["bce", "focal_balanced"],
                   help="'bce' = legacy BCEWithLogitsLoss+pos_weight; "
                        "'focal_balanced' = focal loss + per-(mirna,cell_id) group sample weight (default)")
    p.add_argument("--focal_gamma", type=float, default=2.0,
                   help="focal loss hardness parameter (only used when --loss_kind=focal_balanced)")
    p.add_argument("--no_ts_features", action="store_true",
                   help="disable the 5-d TargetScan hand-crafted feature head (default: enabled)")
    # ----- V4 additions -----
    p.add_argument("--utr_window_len", type=int, default=4000,
                   help="V4: seed-anchored UTR window length (default 4000; V3 used 2000)")
    p.add_argument("--no_aux_seed_head", action="store_true",
                   help="V4: disable the per-position seed-match aux supervision head")
    p.add_argument("--aux_seed_weight", type=float, default=0.3,
                   help="V4: weight on auxiliary seed-mask BCE loss (default 0.3)")
    p.add_argument("--no_convnext", action="store_true",
                   help="V4: disable ConvNeXt UTR encoder (fall back to V3 vanilla CNN)")
    p.add_argument("--train_mixed_only", action="store_true",
                   help="V4 diagnostic: filter training rows to mixed pairs only "
                        "(pairs whose label flips across cells). Same logic as the V3 "
                        "mixed-only diagnostic; off by default.")
    a = p.parse_args()
    holdout_list = None
    if a.holdout_cell_types.strip():
        holdout_list = [s.strip() for s in a.holdout_cell_types.split(",") if s.strip()]
    val_holdout_list = None
    if a.val_holdout_cell_types.strip():
        val_holdout_list = [s.strip() for s in a.val_holdout_cell_types.split(",") if s.strip()]
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
        utr_holdout_frac=float(a.utr_holdout_frac),
        mirna_holdout_frac=float(a.mirna_holdout_frac),
        val_holdout_cell_types=val_holdout_list,
        loss_kind=str(a.loss_kind),
        focal_gamma=float(a.focal_gamma),
        use_ts_features=(not a.no_ts_features),
        # ----- V4 additions -----
        utr_window_len=int(a.utr_window_len),
        use_aux_seed_head=(not a.no_aux_seed_head),
        aux_seed_weight=float(a.aux_seed_weight),
        use_convnext=(not a.no_convnext),
        train_mixed_only=bool(a.train_mixed_only),
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

    if not cache_exists(CACHE_DIR, expected_seed_rule=cfg.seed_rule,
                        expected_utr_holdout_frac=cfg.utr_holdout_frac,
                        expected_mirna_holdout_frac=cfg.mirna_holdout_frac,
                        expected_holdout_cell_types=cfg.holdout_cell_types,
                        expected_val_holdout_cell_types=cfg.val_holdout_cell_types):
        logger.log(f"[phase1] building cache from scratch (seed_rule={cfg.seed_rule} "
                   f"utr_holdout={cfg.utr_holdout_frac} mirna_holdout={cfg.mirna_holdout_frac})")
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
            utr_holdout_frac=cfg.utr_holdout_frac,
            mirna_holdout_frac=cfg.mirna_holdout_frac,
            val_holdout_cell_types=cfg.val_holdout_cell_types,
            seed=cfg.seed,
        )
        splits = make_paper_split(master, split_cfg, gmaps["cell_type_to_idx"], logger)
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

    # V4: forward the user-chosen UTR window length + seed-mask flag into the dataset
    _ds_utr_window = int(cfg.utr_window_len)
    _ds_build_mask = bool(cfg.use_aux_seed_head)
    _ds_mask_ds = UTR_DOWNSAMPLE_V4  # encoder total stride; matches make_model UTR_TOTAL_STRIDE
    logger.log(f"[setup] V4 dataset: UTR_WINDOW_LEN={_ds_utr_window} "
               f"build_seed_mask={_ds_build_mask} downsample={_ds_mask_ds}")

    # ---- Optional V4 diagnostic: mixed-pair-only train filter ----
    # Applied before building datasets, so the filtered train rows propagate
    # into LazyMTIDataset, sample_weight, baseline, everything.
    mixed_only_baselines: dict[str, float] | None = None
    if cfg.train_mixed_only:
        logger.log(">>> RUNNING IN MIXED-PAIR-ONLY DIAGNOSTIC MODE <<<")
        mixed_only_baselines = filter_train_to_mixed_only(cache, logger)

    def _make_ds(name: str):
        return LazyMTIDataset(
            rows=cache["splits"][name],
            mirna_codes_global=cache["mirna_codes_global"],
            seed_targets=cache["seed_targets"],
            utr_seqs=cache["utr_seqs"],
            ts_feat_dict=cache.get("ts_feat_dict"),
            ts_feat_means=cache.get("ts_feat_means"),
            ts_feat_stds=cache.get("ts_feat_stds"),
            utr_window_len=_ds_utr_window,
            build_seed_mask=_ds_build_mask,
            seed_mask_downsample=_ds_mask_ds,
        )

    train_ds = _make_ds("train")
    val_ds = _make_ds("val")
    # Per-epoch validation datasets (used for early stopping)
    per_epoch_val_names = ["val_utr_holdout", "val_cell_holdout", "val_utr_x_cell_holdout"]
    per_epoch_val_dss: dict[str, LazyMTIDataset] = {}
    for n in per_epoch_val_names:
        if n in cache["splits"] and len(cache["splits"][n]["label"]) > 0:
            per_epoch_val_dss[n] = _make_ds(n)
    # Final-eval-only datasets (touched once at the end)
    eval_names = [
        "test_triple",
        "test_cell_holdout", "test_pair_x_cell_holdout",
        "test_utr_holdout", "test_utr_x_cell_holdout",
        "test_mirna_holdout", "test_mirna_x_cell_holdout",
    ]
    eval_dss: dict[str, LazyMTIDataset] = {}
    for n in eval_names:
        if n in cache["splits"] and len(cache["splits"][n]["label"]) > 0:
            eval_dss[n] = _make_ds(n)
    logger.log(
        f"[data] train={len(train_ds):,} val={len(val_ds):,} "
        + " ".join(f"{n}={len(ds):,}" for n, ds in per_epoch_val_dss.items())
        + " | final_eval: "
        + " ".join(f"{n}={len(ds):,}" for n, ds in eval_dss.items())
    )

    # Load UTR-only baselines for normalized-lift composite early stopping.
    # In mixed-only mode, override with baselines recomputed from the filtered train.
    es_baselines_path = CACHE_DIR / "early_stop_baselines.json"
    es_baselines: dict[str, float] = {}
    if es_baselines_path.exists():
        es_baselines = json.loads(es_baselines_path.read_text(encoding="utf-8"))
    if mixed_only_baselines is not None:
        es_baselines = mixed_only_baselines
        logger.log(f"[early_stop] using MIXED-ONLY UTR baselines: {es_baselines}")
        (run_dir / "early_stop_baselines_mixed_only.json").write_text(
            json.dumps(es_baselines, indent=2), encoding="utf-8")
    else:
        logger.log(f"[early_stop] UTR-only baselines: {es_baselines}")

    # ---- Phase 4: model + optimiser + loss ----
    model = make_model(n_cell_types, cfg.model_kind, cfg.cell_embed_dim,
                       use_ts_features=cfg.use_ts_features,
                       ts_feature_dim=TARGETSCAN_FEATURE_DIM,
                       use_convnext=cfg.use_convnext,
                       use_aux_seed_head=cfg.use_aux_seed_head,
                       convnext_dims=cfg.convnext_dims,
                       convnext_depths=cfg.convnext_depths).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"[model] V4 kind={cfg.model_kind} convnext={cfg.use_convnext} "
               f"aux_seed_head={cfg.use_aux_seed_head} aux_weight={cfg.aux_seed_weight} "
               f"trainable_params={n_params:,}")

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

    if cfg.loss_kind == "bce":
        _bce_legacy = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        def loss_fn(logits, y, weight):  # noqa: E306
            return _bce_legacy(logits, y)
        logger.log("[loss] kind=bce (legacy BCEWithLogitsLoss + pos_weight)")
    else:  # focal_balanced
        _pw_scalar = float(pw_val if pos_weight_tensor is not None else 1.0)
        _gamma = float(cfg.focal_gamma)
        def loss_fn(logits, y, weight):  # noqa: E306
            return weighted_focal_loss(logits, y, weight,
                                       gamma=_gamma, pos_weight_val=_pw_scalar)
        logger.log(f"[loss] kind=focal_balanced (gamma={_gamma}, alpha_pos={_pw_scalar/(1+_pw_scalar):.3f}, "
                   f"per-(mirna,cell) group sample_weight)")

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
    per_epoch_val_loaders = {
        n: DataLoader(ds, batch_size=cfg.batch_size * 2, shuffle=False,
                      num_workers=cfg.num_workers, collate_fn=collate_batch,
                      pin_memory=pin,
                      persistent_workers=(cfg.num_workers > 0))
        for n, ds in per_epoch_val_dss.items()
    }

    def _compute_composite(metrics: dict) -> float:
        """Normalized-lift early-stop score: (m - baseline) / (1 - baseline) on
        val_utr_holdout ONLY. This split mirrors test_utr_holdout (the main
        per-sequence generalization metric) and has the largest n among val
        sets, so the signal is the most stable.

        We considered also adding val_utr_x_cell_holdout to this composite (it
        mirrors the harder test_utr_x_cell_holdout) but it has materially fewer
        rows, so its noise floor was hurting the early-stop signal. It is still
        evaluated every epoch (see per_epoch_val_names below) and recorded in
        history.json for monitoring — we expect it to move in the same
        direction as val_utr_holdout when the model is genuinely learning
        sequence-from-UTR patterns. val_cell_holdout is also logged but excluded
        from the composite (its baseline is ~0.7-0.9 because its pairs are seen
        in train, so it doesn't reflect the paper's main claim)."""
        score = 0.0
        n_terms = 0
        for n in ("val_utr_holdout",):
            if n not in metrics or n not in es_baselines:
                continue
            m = metrics[n]
            b = es_baselines[n]
            denom = max(1.0 - b, 1e-3)
            score += (m - b) / denom
            n_terms += 1
        return score

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
                                     logger, cfg.log_every_batches, epoch,
                                     aux_weight=(cfg.aux_seed_weight if cfg.use_aux_seed_head else 0.0),
                                     utr_downsample=UTR_DOWNSAMPLE_V4)
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
        # Evaluate per-epoch holdout validation sets for early stopping signal
        per_epoch_metrics = {}
        for n, loader in per_epoch_val_loaders.items():
            ev = evaluate(model, loader, device)
            per_epoch_metrics[n] = ev["pr_auc"]
            row[f"{n}_pr_auc"] = ev["pr_auc"]
            row[f"{n}_roc_auc"] = ev["roc_auc"]
        composite = _compute_composite(per_epoch_metrics)
        row["composite_lift"] = composite
        history.append(row)
        per_epoch_str = " ".join(f"{n}={per_epoch_metrics[n]:.4f}" for n in per_epoch_metrics)
        logger.log(
            f"[epoch {epoch}] train_loss={train_loss:.4f} val_loss={val['val_loss_mean']:.4f} "
            f"val_PR-AUC={val['pr_auc']:.4f} | {per_epoch_str} | "
            f"composite_lift={composite:.4f} elapsed={elapsed:.1f}s"
        )

        # Use composite_lift for early stopping (aligns with paper main metric)
        improved = composite > best_pr_auc + 1e-6
        if improved:
            best_pr_auc = composite
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
                "per_epoch_val_metrics": per_epoch_metrics,
                "composite_lift": composite,
                "config": cfg.__dict__,
            }, run_dir / "best_model.pt")
            logger.log(f"[best] new best at epoch {epoch} composite_lift={best_pr_auc:.4f}")
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

        scheduler.step(composite)
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
