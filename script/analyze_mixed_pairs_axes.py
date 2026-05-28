#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-axis EDA of mixed pairs in a cached split.

A "mixed pair" is a (miRNA, UTR) pair whose labels are not all equal
across the cells it appears in (cell-dependent). This script answers:

  Axis 1 (miRNA):  Which miRNAs contribute most to the mixed-pair pool?
                   Is the cell-aware signal concentrated in a few miRNAs,
                   or spread broadly?

  Axis 2 (UTR):    Same question for UTRs. Are mixed pairs dominated by
                   a handful of "flippable" UTRs?

  Axis 3 (cell):   Within mixed pairs, do certain cells systematically
                   "flip" the label vs the pair's majority? (cell-flip
                   rate: how often this cell's label disagrees with the
                   most common label for the same pair across other cells.)

Outputs:
  - prints 3 ranked top-15 tables (one per axis) + summary stats
  - writes 3 CSVs to runs/_diagnostics/:
      mixed_axis_mirna_<split>.csv
      mixed_axis_utr_<split>.csv
      mixed_axis_cell_<split>.csv

Usage:
    python script/analyze_mixed_pairs_axes.py
    python script/analyze_mixed_pairs_axes.py --split val
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
V3_ROOT = SCRIPT_DIR.parent
CACHE_DIR = V3_ROOT / "cache"


def _section(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}")


def _hist_ascii(values: list[int], bins: list[int], width: int = 40) -> None:
    counts = [0] * (len(bins) - 1)
    for v in values:
        for i in range(len(bins) - 1):
            if bins[i] <= v < bins[i + 1]:
                counts[i] += 1
                break
    mx = max(counts) if counts else 1
    for i, c in enumerate(counts):
        bar = "#" * max(0, c * width // max(mx, 1))
        label = f"[{bins[i]}, {bins[i+1]})"
        print(f"  {label:>12}: {c:>5}  {bar}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="train")
    p.add_argument("--cache_dir", default=str(CACHE_DIR))
    p.add_argument("--out_dir", default=None)
    a = p.parse_args()

    cache_dir = Path(a.cache_dir)
    out_dir = Path(a.out_dir) if a.out_dir else V3_ROOT / "runs" / "_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    split_path = cache_dir / "splits" / f"{a.split}_rows.npz"
    if not split_path.exists():
        print(f"[error] {split_path} not found", file=sys.stderr); sys.exit(1)

    d = np.load(split_path, allow_pickle=True)
    mirna = d["mirna_gid"].astype(int)
    utr = d["utr_gid"].astype(int)
    cell = d["cell_id"].astype(str)
    cell_type = d["cell_type"].astype(str) if "cell_type" in d.files else None
    label = d["label"].astype(int)
    n_rows = len(label)
    print(f"[load] {a.split}: {n_rows:,} rows")

    # mirna seqs
    mirna_seqs: list[str] = []
    p_ms = cache_dir / "global_maps" / "mirna_seqs.json"
    if p_ms.exists():
        mirna_seqs = json.loads(p_ms.read_text(encoding="utf-8"))
    # utr seqs (for length)
    utr_lens: dict[int, int] = {}
    p_us = cache_dir / "global_maps" / "utr_seqs.pkl"
    if p_us.exists():
        with open(p_us, "rb") as fh:
            us = pickle.load(fh)
        utr_lens = {i: len(s) for i, s in enumerate(us)}

    # ----- Find mixed pairs -----
    pair_labels: dict[tuple[int, int], list[int]] = defaultdict(list)
    pair_cells: dict[tuple[int, int], list[str]] = defaultdict(list)
    pair_celltype: dict[tuple[int, int], list[str]] = defaultdict(list)
    pair_row_idx: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, (m, u, c, y) in enumerate(zip(mirna, utr, cell, label)):
        k = (int(m), int(u))
        pair_labels[k].append(int(y))
        pair_cells[k].append(str(c))
        pair_celltype[k].append(str(cell_type[i]) if cell_type is not None else "")
        pair_row_idx[k].append(i)

    mixed_pairs = [k for k, ys in pair_labels.items() if len(set(ys)) > 1]
    n_total = len(pair_labels)
    n_mixed = len(mixed_pairs)
    print(f"[stats] total pairs: {n_total:,}, mixed pairs: {n_mixed:,} ({n_mixed / max(n_total, 1):.1%})")

    # ----- AXIS 1: miRNA -----
    _section(f"AXIS 1 — miRNA (n_mixed_pairs per miRNA)")
    per_mirna_n = Counter()
    per_mirna_purity: dict[int, list[float]] = defaultdict(list)
    per_mirna_rows = Counter()
    for (m, u) in mixed_pairs:
        per_mirna_n[m] += 1
        ys = pair_labels[(m, u)]
        per_mirna_purity[m].append(float(np.std(ys, ddof=0)))
        per_mirna_rows[m] += len(ys)

    print(f"  unique miRNAs contributing mixed pairs: {len(per_mirna_n):,}")
    print(f"  mean mixed pairs per miRNA: {np.mean(list(per_mirna_n.values())):.2f}")
    print(f"  median: {int(np.median(list(per_mirna_n.values())))}")
    print(f"  max:    {max(per_mirna_n.values())}")
    print(f"\n  Distribution histogram (n mixed pairs per miRNA):")
    _hist_ascii(list(per_mirna_n.values()), [0, 1, 2, 4, 8, 16, 32, 64, 128, 1000])

    print(f"\n  TOP 15 miRNAs by number of mixed pairs:")
    print(f"  {'rank':>4} {'mirna_gid':>10} {'n_mixed':>8} {'n_rows':>8} {'mean_purity':>12} {'mirna_seq_first10':>20}")
    rows_sorted = per_mirna_n.most_common(15)
    rows_axis1: list[dict] = []
    for rk, (m, n_mp) in enumerate(rows_sorted, 1):
        purity = float(np.mean(per_mirna_purity[m]))
        seq = mirna_seqs[m][:10] if m < len(mirna_seqs) else ""
        print(f"  {rk:>4} {m:>10} {n_mp:>8} {per_mirna_rows[m]:>8} {purity:>12.3f} {seq:>20}")
    for (m, n_mp) in per_mirna_n.most_common():
        rows_axis1.append({
            "mirna_gid": m, "n_mixed_pairs": n_mp,
            "n_mixed_rows": per_mirna_rows[m],
            "mean_purity": float(np.mean(per_mirna_purity[m])),
            "mirna_seq": mirna_seqs[m] if m < len(mirna_seqs) else "",
        })

    # ----- AXIS 2: UTR -----
    _section(f"AXIS 2 — UTR (n_mixed_pairs per UTR)")
    per_utr_n = Counter()
    per_utr_purity: dict[int, list[float]] = defaultdict(list)
    per_utr_rows = Counter()
    for (m, u) in mixed_pairs:
        per_utr_n[u] += 1
        ys = pair_labels[(m, u)]
        per_utr_purity[u].append(float(np.std(ys, ddof=0)))
        per_utr_rows[u] += len(ys)

    print(f"  unique UTRs contributing mixed pairs: {len(per_utr_n):,}")
    print(f"  mean mixed pairs per UTR: {np.mean(list(per_utr_n.values())):.2f}")
    print(f"  median: {int(np.median(list(per_utr_n.values())))}")
    print(f"  max:    {max(per_utr_n.values())}")
    print(f"\n  Distribution histogram (n mixed pairs per UTR):")
    _hist_ascii(list(per_utr_n.values()), [0, 1, 2, 4, 8, 16, 32, 64, 128, 1000])

    print(f"\n  TOP 15 UTRs by number of mixed pairs:")
    print(f"  {'rank':>4} {'utr_gid':>8} {'n_mixed':>8} {'n_rows':>8} {'mean_purity':>12} {'utr_len':>8}")
    rows_axis2: list[dict] = []
    for rk, (u, n_mp) in enumerate(per_utr_n.most_common(15), 1):
        purity = float(np.mean(per_utr_purity[u]))
        L = utr_lens.get(u, 0)
        print(f"  {rk:>4} {u:>8} {n_mp:>8} {per_utr_rows[u]:>8} {purity:>12.3f} {L:>8}")
    for (u, n_mp) in per_utr_n.most_common():
        rows_axis2.append({
            "utr_gid": u, "n_mixed_pairs": n_mp,
            "n_mixed_rows": per_utr_rows[u],
            "mean_purity": float(np.mean(per_utr_purity[u])),
            "utr_len": utr_lens.get(u, 0),
        })

    # ----- AXIS 3: cell -----
    _section(f"AXIS 3 — cell (involvement in mixed-pair rows + flip rate)")
    # For each cell, count rows in mixed pairs, and how often this cell's
    # label disagrees with the majority of OTHER cells for the same pair.
    cell_rows_total = Counter()
    cell_rows_in_mixed = Counter()
    cell_label_dist: dict[str, list[int]] = defaultdict(list)  # labels in mixed-pair rows
    cell_flip_count = Counter()
    cell_evaluated = Counter()
    cell_type_of: dict[str, str] = {}

    for i, c in enumerate(cell):
        cell_rows_total[c] += 1
        if cell_type is not None:
            cell_type_of[c] = cell_type[i]

    for (m, u) in mixed_pairs:
        ys = pair_labels[(m, u)]
        cs = pair_cells[(m, u)]
        for c, y in zip(cs, ys):
            cell_rows_in_mixed[c] += 1
            cell_label_dist[c].append(y)
            # "flip" = this cell's label != majority of other cells for this pair
            other = [yy for cc, yy in zip(cs, ys) if cc != c]
            if not other:
                continue
            maj = int(round(np.mean(other)))
            cell_evaluated[c] += 1
            if y != maj:
                cell_flip_count[c] += 1

    print(f"  unique cells: {len(cell_rows_total):,}")
    print(f"  cells appearing in any mixed-pair row: {len(cell_rows_in_mixed):,}")

    print(f"\n  TOP 15 cells by n_rows_in_mixed_pairs:")
    print(f"  {'rank':>4} {'cell_id':>16} {'cell_type':>12} {'n_total':>8} "
          f"{'n_in_mixed':>11} {'pos_rate':>9} {'flip_rate':>10}")
    rows_axis3: list[dict] = []
    for rk, (c, n_in_mx) in enumerate(cell_rows_in_mixed.most_common(15), 1):
        n_total_c = cell_rows_total[c]
        labels_in_mixed = cell_label_dist[c]
        pos_rate = float(np.mean(labels_in_mixed)) if labels_in_mixed else 0.0
        flip = cell_flip_count[c] / max(cell_evaluated[c], 1)
        ct = cell_type_of.get(c, "")
        print(f"  {rk:>4} {c:>16} {ct:>12} {n_total_c:>8} "
              f"{n_in_mx:>11} {pos_rate:>9.3f} {flip:>10.3f}")
    for (c, n_in_mx) in cell_rows_in_mixed.most_common():
        labels_in_mixed = cell_label_dist[c]
        rows_axis3.append({
            "cell_id": c,
            "cell_type": cell_type_of.get(c, ""),
            "n_rows_total": cell_rows_total[c],
            "n_rows_in_mixed_pairs": n_in_mx,
            "pos_rate_in_mixed": float(np.mean(labels_in_mixed)) if labels_in_mixed else 0.0,
            "flip_rate": cell_flip_count[c] / max(cell_evaluated[c], 1),
            "n_evaluated_for_flip": cell_evaluated[c],
        })

    # Aggregate by cell_type
    if cell_type is not None:
        _section(f"AXIS 3b — aggregate by cell_type")
        ct_in_mixed = Counter()
        ct_total = Counter()
        ct_pos = Counter()
        for c, n_in_mx in cell_rows_in_mixed.items():
            ct = cell_type_of.get(c, "?")
            ct_in_mixed[ct] += n_in_mx
            ct_total[ct] += cell_rows_total[c]
            ct_pos[ct] += sum(cell_label_dist[c])
        print(f"  {'cell_type':>14} {'n_total':>10} {'n_in_mixed':>12} "
              f"{'%_in_mixed':>11} {'pos_rate_in_mixed':>17}")
        for ct in sorted(ct_in_mixed.keys()):
            pct = ct_in_mixed[ct] / max(ct_total[ct], 1)
            posr = ct_pos[ct] / max(ct_in_mixed[ct], 1)
            print(f"  {ct:>14} {ct_total[ct]:>10,} {ct_in_mixed[ct]:>12,} "
                  f"{pct:>10.1%} {posr:>17.3f}")

    # ----- Save CSVs -----
    def _write_csv(path: Path, rows: list[dict], cols: list[str]):
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            for r in rows:
                w.writerow([r.get(c, "") for c in cols])

    p1 = out_dir / f"mixed_axis_mirna_{a.split}.csv"
    _write_csv(p1, rows_axis1, ["mirna_gid", "n_mixed_pairs", "n_mixed_rows",
                                  "mean_purity", "mirna_seq"])
    p2 = out_dir / f"mixed_axis_utr_{a.split}.csv"
    _write_csv(p2, rows_axis2, ["utr_gid", "n_mixed_pairs", "n_mixed_rows",
                                  "mean_purity", "utr_len"])
    p3 = out_dir / f"mixed_axis_cell_{a.split}.csv"
    _write_csv(p3, rows_axis3, ["cell_id", "cell_type", "n_rows_total",
                                  "n_rows_in_mixed_pairs", "pos_rate_in_mixed",
                                  "flip_rate", "n_evaluated_for_flip"])
    print(f"\n[ok] axis CSVs written:")
    print(f"     {p1}")
    print(f"     {p2}")
    print(f"     {p3}")
    print()


if __name__ == "__main__":
    main()
