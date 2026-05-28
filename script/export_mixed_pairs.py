#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export mixed (miRNA, UTR) pairs from a cached split to CSV for supervisor review.

A pair is "mixed" iff its labels across cells are not all the same value
(i.e. std(labels_across_cells) > 0). These are the cell-dependent pairs
that constitute the entire theoretical gain space for a cell-aware model
over a cell-agnostic baseline. In our train set, mixed pairs ~ 16%.

Outputs two CSVs to `runs/_diagnostics/` (filenames clarified for sharing):

  (A) <split>data_mixedpairs_summary.csv   one row per mixed pair
        columns:
          mirna_gid, utr_gid, n_rows, n_cells, n_pos, n_neg,
          pos_rate, purity, mirna_seq, utr_len, utr_seq_first50

  (B) <split>data_mixedpairs.csv           one row per observation of a mixed pair
        ★ this is the main file (what a supervisor typically asks to see).
        columns:
          mirna_gid, utr_gid, cell_id, cell_type, label,
          mirna_seq, utr_len

Usage:
    python script/export_mixed_pairs.py
    python script/export_mixed_pairs.py --split val
    python script/export_mixed_pairs.py --include_full_utr   # also dump full UTR seq
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
V3_ROOT = SCRIPT_DIR.parent
CACHE_DIR = V3_ROOT / "cache"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="train",
                   help="which cached split to export (default: train)")
    p.add_argument("--cache_dir", default=str(CACHE_DIR))
    p.add_argument("--out_dir", default=None,
                   help="output dir (default: <V3>/runs/_diagnostics/)")
    p.add_argument("--include_full_utr", action="store_true",
                   help="include the full UTR sequence in summary CSV "
                        "(can be very long, default off)")
    p.add_argument("--max_mixed_pairs", type=int, default=0,
                   help="if > 0, limit number of mixed pairs exported "
                        "(useful for sending a small sample to supervisor)")
    args = p.parse_args()

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir) if args.out_dir else V3_ROOT / "runs" / "_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    split_path = cache_dir / "splits" / f"{args.split}_rows.npz"
    if not split_path.exists():
        print(f"[error] split file not found: {split_path}", file=sys.stderr)
        sys.exit(1)

    # ----- Load split rows -----
    data = np.load(split_path, allow_pickle=True)
    mirna = data["mirna_gid"].astype(int)
    utr = data["utr_gid"].astype(int)
    cell = data["cell_id"].astype(str)
    cell_type = data["cell_type"].astype(str) if "cell_type" in data.files else None
    label = data["label"].astype(int)
    n_rows = len(label)
    print(f"[load] {args.split}: {n_rows:,} rows")

    # ----- Load global maps -----
    mirna_seqs: list[str] = []
    mirna_seqs_path = cache_dir / "global_maps" / "mirna_seqs.json"
    if mirna_seqs_path.exists():
        mirna_seqs = json.loads(mirna_seqs_path.read_text(encoding="utf-8"))
        print(f"[load] miRNA seqs: {len(mirna_seqs):,}")

    utr_seqs: list[str] = []
    utr_seqs_path = cache_dir / "global_maps" / "utr_seqs.pkl"
    if utr_seqs_path.exists():
        with open(utr_seqs_path, "rb") as fh:
            utr_seqs = pickle.load(fh)
        print(f"[load] UTR seqs: {len(utr_seqs):,} (median length: "
              f"{int(np.median([len(s) for s in utr_seqs])):,} nt)")

    # ----- Identify mixed pairs -----
    pair_labels: dict[tuple[int, int], list[int]] = defaultdict(list)
    pair_cells: dict[tuple[int, int], list[str]] = defaultdict(list)
    pair_row_indices: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, (m, u, c, y) in enumerate(zip(mirna, utr, cell, label)):
        k = (int(m), int(u))
        pair_labels[k].append(int(y))
        pair_cells[k].append(str(c))
        pair_row_indices[k].append(i)

    mixed_pairs: list[tuple[int, int]] = []
    for k, ys in pair_labels.items():
        if len(set(ys)) > 1:  # both 0 and 1 present
            mixed_pairs.append(k)
    mixed_pairs.sort(key=lambda k: -len(pair_labels[k]))  # most rows first

    n_total = len(pair_labels)
    n_mixed = len(mixed_pairs)
    print(f"\n[stats] total pairs:  {n_total:,}")
    print(f"[stats] mixed pairs:  {n_mixed:,}  ({n_mixed / max(n_total, 1):.1%})")

    if args.max_mixed_pairs > 0:
        mixed_pairs = mixed_pairs[:args.max_mixed_pairs]
        print(f"[stats] limited to top {args.max_mixed_pairs:,} mixed pairs by n_rows")

    # ----- (A) Summary CSV: one row per mixed pair -----
    summary_path = out_dir / f"{args.split}data_mixedpairs_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        header = ["mirna_gid", "utr_gid", "n_rows", "n_cells",
                  "n_pos", "n_neg", "pos_rate", "purity",
                  "mirna_seq", "utr_len"]
        if args.include_full_utr:
            header.append("utr_seq")
        else:
            header.append("utr_seq_first50")
        w.writerow(header)

        for (m, u) in mixed_pairs:
            ys = pair_labels[(m, u)]
            cs = pair_cells[(m, u)]
            n = len(ys)
            n_pos = sum(ys)
            n_neg = n - n_pos
            pos_rate = n_pos / max(n, 1)
            purity = float(np.std(ys, ddof=0))
            mirna_seq = mirna_seqs[m] if m < len(mirna_seqs) else ""
            utr_seq = utr_seqs[u] if u < len(utr_seqs) else ""
            utr_field = utr_seq if args.include_full_utr else utr_seq[:50]
            row = [m, u, n, len(set(cs)), n_pos, n_neg,
                   f"{pos_rate:.4f}", f"{purity:.4f}",
                   mirna_seq, len(utr_seq), utr_field]
            w.writerow(row)
    print(f"\n[ok] summary CSV written: {summary_path}")
    print(f"     {len(mixed_pairs):,} rows (one row per mixed pair)")

    # ----- (B) Long CSV: one row per observation of a mixed pair (main file) -----
    long_path = out_dir / f"{args.split}data_mixedpairs.csv"
    n_long_rows = 0
    with long_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["mirna_gid", "utr_gid", "cell_id", "cell_type", "label",
                    "mirna_seq", "utr_len"])
        for (m, u) in mixed_pairs:
            mirna_seq = mirna_seqs[m] if m < len(mirna_seqs) else ""
            utr_len = len(utr_seqs[u]) if u < len(utr_seqs) else 0
            for idx in pair_row_indices[(m, u)]:
                ct = cell_type[idx] if cell_type is not None else ""
                w.writerow([m, u, cell[idx], ct, label[idx],
                            mirna_seq, utr_len])
                n_long_rows += 1
    print(f"[ok] long CSV written:    {long_path}")
    print(f"     {n_long_rows:,} rows (one row per observation of a mixed pair)")
    print()

    # ----- Quick sanity print: top 5 mixed pairs -----
    print(f"[preview] top 5 mixed pairs (most rows):")
    print(f"  {'mirna':>6} {'utr':>6} {'n_rows':>8} {'n_cells':>8} "
          f"{'n_pos':>6} {'n_neg':>6} {'pos_rate':>9} {'purity':>7}")
    for (m, u) in mixed_pairs[:5]:
        ys = pair_labels[(m, u)]
        cs = pair_cells[(m, u)]
        n = len(ys); n_pos = sum(ys); n_neg = n - n_pos
        print(f"  {m:>6} {u:>6} {n:>8} {len(set(cs)):>8} "
              f"{n_pos:>6} {n_neg:>6} {n_pos / n:>9.3f} "
              f"{float(np.std(ys, ddof=0)):>7.3f}")
    print()


if __name__ == "__main__":
    main()
