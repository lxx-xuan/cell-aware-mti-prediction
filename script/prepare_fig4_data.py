#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare CSVs for Figure 4.1 (dataset structural analysis).

Reads:  cache/splits/<split>_rows.npz
Writes: runs/_diagnostics/
    --- raw (for matplotlib `plot_fig4_dataset.py`) ---
        fig4a_per_mirna_pos_rate.csv         one row per miRNA   (FULL train)
        fig4b_per_utr_pos_rate.csv           one row per UTR     (FULL train)
        fig4c_per_cell_pos_rate_mixed.csv    one row per cell_id (MIXED only)
    --- pre-binned (for BR Column chart, since BR has no histogram) ---
        fig4a_binned_for_BR.csv              40 bins of (a)
        fig4b_binned_for_BR.csv              40 bins of (b)

Usage in BioRender Graphing:
  (a) Load fig4a_binned_for_BR.csv → Column chart →
      X = bin_center, Y = count, Color by `category` (narrow / other)
  (b) Load fig4b_binned_for_BR.csv → Column chart →
      X = bin_center, Y = count, Color by `category` (extreme / other)
  (c) Load fig4c_per_cell_pos_rate_mixed.csv → Box plot →
      X = cell_type, Y = pos_rate_in_mixed, Color by `lineage`,
      Analysis: Two-Way ANOVA + Tukey on lineage groups

CLI:
    python script/prepare_fig4_data.py
    python script/prepare_fig4_data.py --split val
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
V3_ROOT = SCRIPT_DIR.parent
CACHE_DIR = V3_ROOT / "cache"

# Lymphoid / myeloid lineage mapping.
# Extend this dict as new cell types appear; unknown -> "other".
LINEAGE_MAP = {
    # ---- lymphoid ----
    "CD4": "lymphoid", "CD4_T": "lymphoid", "CD4 T": "lymphoid",
    "CD8": "lymphoid", "CD8a": "lymphoid", "CD8n": "lymphoid", "CD8 T": "lymphoid",
    "B": "lymphoid", "B1": "lymphoid", "B2": "lymphoid",
    "plasma": "lymphoid", "Plasma": "lymphoid", "plasmablast": "lymphoid",
    "NK": "lymphoid", "NK1": "lymphoid", "NK2": "lymphoid", "NKT": "lymphoid",
    "Tregs": "lymphoid", "Treg": "lymphoid",
    "Tdg": "lymphoid",   # γδ T
    "ILC": "lymphoid",
    # ---- myeloid ----
    "Gran": "myeloid", "granulocyte": "myeloid", "neutrophil": "myeloid",
    "Mono": "myeloid", "mono": "myeloid", "monocyte": "myeloid",
    "ncMono": "myeloid", "Mono_CD14": "myeloid", "Mono_CD16": "myeloid",
    "cDC": "myeloid", "cDC1": "myeloid", "cDC2": "myeloid",
    "pDC": "myeloid",
    "mDC": "myeloid", "mDC1": "myeloid", "mDC2": "myeloid",
    "macrophage": "myeloid", "Macrophage": "myeloid", "Mac": "myeloid",
}


def lineage_of(ct: str) -> str:
    if not ct:
        return "other"
    if ct in LINEAGE_MAP:
        return LINEAGE_MAP[ct]
    low = ct.lower()
    for key, lin in LINEAGE_MAP.items():
        if key.lower() == low:
            return lin
    return "other"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="train")
    p.add_argument("--cache_dir", default=str(CACHE_DIR))
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir) if args.out_dir else V3_ROOT / "runs" / "_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    split_path = cache_dir / "splits" / f"{args.split}_rows.npz"
    if not split_path.exists():
        print(f"[error] split file not found: {split_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[load] {split_path}")
    d = np.load(split_path, allow_pickle=True)
    mirna = d["mirna_gid"].astype(int)
    utr = d["utr_gid"].astype(int)
    cell = d["cell_id"].astype(str)
    cell_type = d["cell_type"].astype(str) if "cell_type" in d.files else None
    label = d["label"].astype(int)
    n_rows = len(label)
    print(f"  rows={n_rows:,}  unique miRNAs={len(set(mirna)):,}  "
          f"UTRs={len(set(utr)):,}  cells={len(set(cell)):,}")
    if cell_type is None:
        print("[warn] cell_type field missing — panel (c) lineage column will be 'other'")

    # ---- (a) per-miRNA pos_rate on FULL train ----
    per_mirna: dict[int, dict] = defaultdict(lambda: {"n": 0, "pos": 0})
    for m, y in zip(mirna, label):
        s = per_mirna[int(m)]
        s["n"] += 1
        s["pos"] += int(y)

    out_a = out_dir / "fig4a_per_mirna_pos_rate.csv"
    with out_a.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["mirna_gid", "n_rows", "pos_rate"])
        for m, s in per_mirna.items():
            pr = s["pos"] / max(s["n"], 1)
            w.writerow([m, s["n"], f"{pr:.6f}"])
    prs_m = np.array([s["pos"] / max(s["n"], 1) for s in per_mirna.values()])
    pct_narrow = float(np.mean((prs_m >= 0.10) & (prs_m <= 0.30)))
    print(f"[save] {out_a.name}  ({len(per_mirna):,} miRNAs)")
    print(f"       in [0.10, 0.30]: {pct_narrow:.1%}   "
          f"median={np.median(prs_m):.3f}  mean={prs_m.mean():.3f}")

    # ---- (b) per-UTR pos_rate on FULL train ----
    per_utr: dict[int, dict] = defaultdict(lambda: {"n": 0, "pos": 0})
    for u, y in zip(utr, label):
        s = per_utr[int(u)]
        s["n"] += 1
        s["pos"] += int(y)

    out_b = out_dir / "fig4b_per_utr_pos_rate.csv"
    with out_b.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["utr_gid", "n_rows", "pos_rate"])
        for u, s in per_utr.items():
            pr = s["pos"] / max(s["n"], 1)
            w.writerow([u, s["n"], f"{pr:.6f}"])
    prs_u = np.array([s["pos"] / max(s["n"], 1) for s in per_utr.values()])
    pct_extreme = float(np.mean((prs_u < 0.05) | (prs_u > 0.95)))
    print(f"[save] {out_b.name}  ({len(per_utr):,} UTRs)")
    print(f"       extreme (<0.05 ∪ >0.95): {pct_extreme:.1%}   "
          f"<0.05: {np.mean(prs_u<0.05):.1%}   >0.95: {np.mean(prs_u>0.95):.1%}")

    # ---- (c) per-cell pos_rate in MIXED pairs ----
    pair_labels: dict[tuple, list] = defaultdict(list)
    pair_cells: dict[tuple, list] = defaultdict(list)
    pair_rowidx: dict[tuple, list] = defaultdict(list)
    for i, (m, u, c, y) in enumerate(zip(mirna, utr, cell, label)):
        k = (int(m), int(u))
        pair_labels[k].append(int(y))
        pair_cells[k].append(c)
        pair_rowidx[k].append(i)

    mixed = [k for k, ys in pair_labels.items() if len(set(ys)) > 1]
    print(f"  mixed pairs: {len(mixed):,} / {len(pair_labels):,} "
          f"({len(mixed) / max(len(pair_labels), 1):.1%})")

    per_cell: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "pos": 0, "ct": ""}
    )
    for k in mixed:
        for c, y, ri in zip(pair_cells[k], pair_labels[k], pair_rowidx[k]):
            s = per_cell[c]
            s["n"] += 1
            s["pos"] += y
            if not s["ct"] and cell_type is not None:
                s["ct"] = str(cell_type[ri])

    out_c = out_dir / "fig4c_per_cell_pos_rate_mixed.csv"
    with out_c.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["cell_id", "cell_type", "lineage",
                    "n_rows_in_mixed", "pos_rate_in_mixed"])
        for c, s in per_cell.items():
            pr = s["pos"] / max(s["n"], 1)
            lin = lineage_of(s["ct"])
            w.writerow([c, s["ct"], lin, s["n"], f"{pr:.6f}"])
    print(f"[save] {out_c.name}  ({len(per_cell):,} cell_ids)")

    # cell_type aggregate summary
    ct_n: dict[str, int] = defaultdict(int)
    ct_pos: dict[str, float] = defaultdict(float)
    for c, s in per_cell.items():
        ct = s["ct"]
        if not ct:
            continue
        ct_n[ct] += s["n"]
        ct_pos[ct] += s["pos"]
    order = {"lymphoid": 0, "myeloid": 1, "other": 2}
    rows = [(ct, ct_pos[ct] / max(ct_n[ct], 1), lineage_of(ct)) for ct in ct_n]
    rows.sort(key=lambda r: (order[r[2]], -r[1]))
    print("\n[summary] cell_type → mean pos_rate in mixed pairs:")
    print(f"  {'cell_type':>14}  {'lineage':>9}  {'pos_rate':>9}  {'n_rows':>10}")
    for ct, pr, lin in rows:
        print(f"  {ct:>14}  {lin:>9}  {pr:>9.3f}  {ct_n[ct]:>10,}")

    # ---- (a-binned) and (b-binned) for BR Column chart ----
    def _bin_for_br(values: np.ndarray, n_bins: int = 40,
                    narrow_range=None, mark_extreme: bool = False) -> list[dict]:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        counts, _ = np.histogram(values, bins=edges)
        rows = []
        for i, c in enumerate(counts):
            lo, hi = float(edges[i]), float(edges[i + 1])
            mid = (lo + hi) / 2
            cat = "other"
            if narrow_range is not None:
                a, b = narrow_range
                if a <= mid <= b:
                    cat = "narrow"
            if mark_extreme and (mid < 0.05 or mid > 0.95):
                cat = "extreme"
            rows.append({
                "bin_left": lo, "bin_right": hi, "bin_center": mid,
                "bin_label": f"{lo:.3f}-{hi:.3f}",
                "count": int(c), "category": cat,
            })
        return rows

    def _save_binned(path: Path, rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["bin_left", "bin_right", "bin_center",
                        "bin_label", "count", "category"])
            for r in rows:
                w.writerow([f"{r['bin_left']:.4f}", f"{r['bin_right']:.4f}",
                            f"{r['bin_center']:.4f}", r["bin_label"],
                            r["count"], r["category"]])

    rows_a_bin = _bin_for_br(prs_m, narrow_range=(0.10, 0.30))
    out_a_bin = out_dir / "fig4a_binned_for_BR.csv"
    _save_binned(out_a_bin, rows_a_bin)
    print(f"[save] {out_a_bin.name}  (40 bins)")

    rows_b_bin = _bin_for_br(prs_u, mark_extreme=True)
    out_b_bin = out_dir / "fig4b_binned_for_BR.csv"
    _save_binned(out_b_bin, rows_b_bin)
    print(f"[save] {out_b_bin.name}  (40 bins)")

    print("\n[done] 5 CSVs written to", out_dir)
    print("       raw → use plot_fig4_dataset.py (matplotlib)")
    print("       binned → load into BR Graphing as Column chart")


if __name__ == "__main__":
    main()
