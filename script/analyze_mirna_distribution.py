#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze per-miRNA label distribution in the cached train set.

Outputs (all printed to stdout):

  (1) Overall summary: n_rows, n_unique miRNAs / UTRs / cells, overall pos_rate.

  (2) Per-miRNA pos_rate histogram. Tells us whether miRNAs are roughly
      homogeneous (all hovering around the global pos_rate) or split into
      "weak suppressor" (pos_rate << 0.2) and "strong suppressor"
      (pos_rate >> 0.2) populations.

  (3) Per-miRNA n_rows distribution percentiles + top-10 / bottom-10 lists.
      Tells us whether a handful of miRNAs dominate the training data
      (long-tailed) or contribution is roughly even.

  (4) Cross-cell label std per miRNA. For each miRNA we compute the
      pos_rate within each cell_id (where the miRNA has >= 3 observations
      in that cell) and report std across cells. High std => same miRNA
      behaves differently across cells => there is genuine cell-context
      signal the model could exploit.

  (5) Per-UTR pos_rate "shortcut score". For each UTR we compute
      pos_rate across all (miRNA, cell) it appears in. The standard
      deviation of these per-UTR pos_rates is a direct measure of how
      predictive UTR identity alone is. High std + bimodal distribution
      => UTR identity is highly informative => UTR-only baseline will
      be high => model can trivially memorise UTR identity (= our
      observed UTR-shortcut).

  (6) Joint pair-purity summary (mirrors §3.2): fraction of (miRNA, UTR)
      pairs that are pure (label invariant across cells) vs mixed.

Optional: also saves a CSV `runs/_diagnostics/per_mirna_distribution.csv`
with per-miRNA stats for follow-up analysis or plotting.

Usage:
    python script/analyze_mirna_distribution.py
    python script/analyze_mirna_distribution.py --split val           # any cached split
    python script/analyze_mirna_distribution.py --no_save_csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
V3_ROOT = SCRIPT_DIR.parent
CACHE_DIR = V3_ROOT / "cache"


def _print_section(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}")


def _ascii_hist(counts: list[int], width: int = 50) -> list[str]:
    if not counts:
        return []
    mx = max(counts)
    if mx == 0:
        return ["" for _ in counts]
    return ["#" * max(1, c * width // mx) if c > 0 else "" for c in counts]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="train",
                   help="which cached split to analyse (default: train; "
                        "any cached split file under cache/splits/ works)")
    p.add_argument("--no_save_csv", action="store_true",
                   help="do not write per-miRNA CSV to runs/_diagnostics/")
    p.add_argument("--cache_dir", default=str(CACHE_DIR),
                   help="path to the cache/ directory (default: <V3>/cache)")
    args = p.parse_args()

    cache_dir = Path(args.cache_dir)
    split_path = cache_dir / "splits" / f"{args.split}_rows.npz"
    if not split_path.exists():
        print(f"[error] split file not found: {split_path}", file=sys.stderr)
        print(f"[error] available splits:", file=sys.stderr)
        for p in sorted((cache_dir / "splits").glob("*_rows.npz")):
            print(f"          {p.name}", file=sys.stderr)
        sys.exit(1)

    # ----- Load -----
    data = np.load(split_path, allow_pickle=True)
    mirna = data["mirna_gid"].astype(int)
    utr = data["utr_gid"].astype(int)
    cell = data["cell_id"].astype(str)
    label = data["label"].astype(int)
    n_rows = len(label)

    # miRNA seqs (for length info; optional)
    mirna_seqs: list[str] = []
    mirna_seqs_path = cache_dir / "global_maps" / "mirna_seqs.json"
    if mirna_seqs_path.exists():
        try:
            mirna_seqs = json.loads(mirna_seqs_path.read_text(encoding="utf-8"))
        except Exception:
            mirna_seqs = []

    # ----- (1) Overall -----
    _print_section(f"OVERALL SUMMARY ({args.split} split)")
    print(f"  rows               : {n_rows:,}")
    print(f"  unique miRNAs      : {len(set(mirna)):,}")
    print(f"  unique UTRs        : {len(set(utr)):,}")
    print(f"  unique cells       : {len(set(cell)):,}")
    print(f"  overall pos_rate   : {label.mean():.4f}")

    # ----- per-miRNA aggregation -----
    per_mirna: dict[int, dict] = defaultdict(
        lambda: {
            "rows": 0,
            "pos": 0,
            "utrs": set(),
            "cells": set(),
            "per_cell_labels": defaultdict(list),
        }
    )
    for m, u, c, y in zip(mirna, utr, cell, label):
        s = per_mirna[int(m)]
        s["rows"] += 1
        s["pos"] += int(y)
        s["utrs"].add(int(u))
        s["cells"].add(str(c))
        s["per_cell_labels"][str(c)].append(int(y))

    # build summary rows
    stats: list[dict] = []
    for m, s in per_mirna.items():
        n = s["rows"]
        pos_rate = s["pos"] / max(n, 1)
        cell_pos_rates = [
            float(np.mean(ys)) for ys in s["per_cell_labels"].values() if len(ys) >= 3
        ]
        cell_std = float(np.std(cell_pos_rates)) if len(cell_pos_rates) >= 2 else 0.0
        stats.append({
            "mirna_gid": m,
            "n_rows": n,
            "n_UTRs": len(s["utrs"]),
            "n_cells": len(s["cells"]),
            "pos_rate": pos_rate,
            "cross_cell_std": cell_std,
            "n_cells_with_3plus": len(cell_pos_rates),
            "mirna_len": len(mirna_seqs[m]) if m < len(mirna_seqs) else 0,
        })
    stats.sort(key=lambda x: -x["n_rows"])

    # ----- (2) pos_rate histogram -----
    _print_section("(2) per-miRNA pos_rate distribution")
    bins = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 1.01]
    counts = [0] * (len(bins) - 1)
    for s in stats:
        pr = s["pos_rate"]
        for i in range(len(bins) - 1):
            if bins[i] <= pr < bins[i + 1]:
                counts[i] += 1
                break
    bars = _ascii_hist(counts, width=50)
    for i, (c, bar) in enumerate(zip(counts, bars)):
        print(f"  [{bins[i]:>5.2f} - {bins[i+1]:>5.2f}): {c:>5}  {bar}")
    print(f"\n  total miRNAs binned: {sum(counts):,}")
    # quick descriptors
    prs = np.array([s["pos_rate"] for s in stats])
    print(f"  mean: {prs.mean():.4f}  median: {np.median(prs):.4f}  "
          f"std: {prs.std():.4f}")
    print(f"  fraction with pos_rate in [0.10, 0.30]: {np.mean((prs >= 0.10) & (prs <= 0.30)):.1%}")
    print(f"  fraction with pos_rate < 0.05:           {np.mean(prs < 0.05):.1%}  (near-all-negative)")
    print(f"  fraction with pos_rate > 0.50:           {np.mean(prs > 0.50):.1%}  (mostly-positive)")

    # ----- (3) n_rows distribution -----
    _print_section("(3) per-miRNA n_rows distribution")
    n_rows_arr = np.array(sorted([s["n_rows"] for s in stats]))
    for pct in [5, 25, 50, 75, 90, 95, 99]:
        idx = max(0, int(len(n_rows_arr) * pct / 100) - 1)
        print(f"  p{pct:>2}: {n_rows_arr[idx]:>9,} rows / miRNA")
    print(f"  total miRNAs: {len(stats):,}, total rows: {int(n_rows_arr.sum()):,}")
    # tail share
    top10pct_idx = max(1, len(stats) // 10)
    top10pct_rows = sum(s["n_rows"] for s in stats[:top10pct_idx])
    print(f"  top 10% miRNAs cover {top10pct_rows:,} rows "
          f"({top10pct_rows / n_rows:.1%} of total)")

    print(f"\n  TOP 10 miRNAs by n_rows:")
    print(f"  {'mirna':>7} {'n_rows':>10} {'n_UTRs':>8} {'n_cells':>8} "
          f"{'pos_rate':>9} {'cc_std':>7}")
    for s in stats[:10]:
        print(f"  {s['mirna_gid']:>7} {s['n_rows']:>10,} {s['n_UTRs']:>8} "
              f"{s['n_cells']:>8} {s['pos_rate']:>9.3f} {s['cross_cell_std']:>7.3f}")

    print(f"\n  BOTTOM 10 miRNAs by n_rows:")
    print(f"  {'mirna':>7} {'n_rows':>10} {'n_UTRs':>8} {'n_cells':>8} "
          f"{'pos_rate':>9} {'cc_std':>7}")
    for s in stats[-10:]:
        print(f"  {s['mirna_gid']:>7} {s['n_rows']:>10,} {s['n_UTRs']:>8} "
              f"{s['n_cells']:>8} {s['pos_rate']:>9.3f} {s['cross_cell_std']:>7.3f}")

    # ----- (4) cross-cell std (cell-conditioning signal) -----
    _print_section("(4) cross-cell label std (cell-conditioning signal)")
    cc = np.array([s["cross_cell_std"] for s in stats if s["n_cells_with_3plus"] >= 2])
    if len(cc):
        print(f"  miRNAs with >= 2 cells (n>=3 obs each):  {len(cc):,}")
        print(f"  mean cross_cell_std:                     {cc.mean():.4f}")
        print(f"  median:                                  {float(np.median(cc)):.4f}")
        print(f"  Fraction with cross_cell_std > 0.05:     {np.mean(cc > 0.05):.1%}")
        print(f"  Fraction with cross_cell_std > 0.10:     {np.mean(cc > 0.10):.1%}")
        print(f"  Fraction with cross_cell_std > 0.20:     {np.mean(cc > 0.20):.1%}")
        print()
        print(f"  → 高 cross_cell_std => 同 miRNA 在不同 cell 上 pos_rate 差异大,")
        print(f"    意味着 cell-conditioning 有 signal 可学; 低 std => cell 几乎不影响该 miRNA.")
    else:
        print(f"  (not enough miRNAs with >=2 cells of >=3 obs each)")

    # ----- (5) per-UTR pos_rate (UTR-shortcut indicator) -----
    _print_section("(5) per-UTR pos_rate (UTR-identity shortcut indicator)")
    per_utr: dict[int, dict] = defaultdict(lambda: {"rows": 0, "pos": 0})
    for u, y in zip(utr, label):
        per_utr[int(u)]["rows"] += 1
        per_utr[int(u)]["pos"] += int(y)
    utr_pr = np.array([s["pos"] / max(s["rows"], 1) for s in per_utr.values()])
    print(f"  unique UTRs:                          {len(utr_pr):,}")
    print(f"  mean of per-UTR pos_rate:             {utr_pr.mean():.4f}")
    print(f"  std of per-UTR pos_rate:              {utr_pr.std():.4f}")
    print(f"  Fraction of UTRs with pos_rate < 0.05 (near-all-negative): {np.mean(utr_pr < 0.05):.1%}")
    print(f"  Fraction of UTRs with pos_rate > 0.95 (near-all-positive): {np.mean(utr_pr > 0.95):.1%}")
    print(f"  Fraction of UTRs with pos_rate in [0.40, 0.60] (balanced): {np.mean((utr_pr >= 0.40) & (utr_pr <= 0.60)):.1%}")
    print()
    print(f"  → std 大 + 大量 UTR 极端 (>0.95 或 <0.05) => UTR identity 高度可预测 label")
    print(f"    => UTR-only baseline 高 => 解释了 V3 在 test_triple/test_cell_holdout 高分.")

    # ----- (6) pair-purity (mirrors §3.2) -----
    _print_section("(6) (miRNA, UTR) pair purity (cell-dependence)")
    pair_labels: dict[tuple, set] = defaultdict(set)
    for m, u, y in zip(mirna, utr, label):
        pair_labels[(int(m), int(u))].add(int(y))
    n_pure = sum(1 for v in pair_labels.values() if len(v) == 1)
    n_mixed = sum(1 for v in pair_labels.values() if len(v) > 1)
    n_total = len(pair_labels)
    print(f"  unique (miRNA, UTR) pairs: {n_total:,}")
    print(f"  pure pairs  (single label across cells): {n_pure:,} ({n_pure / max(n_total, 1):.1%})")
    print(f"  mixed pairs (label flips across cells):  {n_mixed:,} ({n_mixed / max(n_total, 1):.1%})")
    print(f"\n  → mixed pair 占比是 cell-aware 模型相对 cell-agnostic 模型的全部理论增益空间.")

    # ----- (7) Save per-miRNA CSV -----
    if not args.no_save_csv:
        out_dir = V3_ROOT / "runs" / "_diagnostics"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"per_mirna_distribution_{args.split}.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["mirna_gid", "mirna_len", "n_rows", "n_UTRs",
                        "n_cells", "n_cells_with_3plus_obs",
                        "pos_rate", "cross_cell_std"])
            for s in stats:
                w.writerow([
                    s["mirna_gid"], s["mirna_len"], s["n_rows"], s["n_UTRs"],
                    s["n_cells"], s["n_cells_with_3plus"],
                    f"{s['pos_rate']:.6f}", f"{s['cross_cell_std']:.6f}",
                ])
        print(f"\n[ok] per-miRNA CSV written: {out_csv}")
    print()


if __name__ == "__main__":
    main()
