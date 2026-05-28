#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig 4.3 — Per-cell PR-AUC heatmap.

Reads a single model's `final_report.json` and renders a heatmap with:
  rows  = 7 test sets (ordered by # axes held out)
  cols  = 12 cell types
  color = PR-AUC

Demonstrates whether the UTR-encoder failure is uniform across cell types
(supporting the "structural bottleneck" interpretation).

Usage:
    python script/plot_fig43_per_cell_heatmap.py \
        --report runs/main_film_seed42/final_report.json \
        --title "V3 full — per-cell PR-AUC"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


SCRIPT_DIR = Path(__file__).resolve().parent
V3_ROOT = SCRIPT_DIR.parent
DIAG_DIR = V3_ROOT / "runs" / "_diagnostics"

# Unified palette anchored on #5B9BD5 (blue) + #F4C430 (yellow)
ANCHOR_BLUE   = "#5B9BD5"
ANCHOR_YELLOW = "#F4C430"
DEEP_BLUE     = "#1F4E79"

# Diverging colormap: yellow (random / failing) → white (mid) → blue (good)
# Best for PR-AUC where ~0.22 = random baseline and ~1.0 = perfect
CMAP_PR = LinearSegmentedColormap.from_list(
    "cmap_pr", [ANCHOR_YELLOW, "#FFFFFF", ANCHOR_BLUE, DEEP_BLUE])

TEST_ORDER = [
    "test_triple",
    "test_mirna_holdout",
    "test_cell_holdout",
    "test_pair_x_cell_holdout",
    "test_mirna_x_cell_holdout",
    "test_utr_holdout",
    "test_utr_x_cell_holdout",
]
TEST_LABELS = {
    "test_triple":               "triple (all seen)",
    "test_mirna_holdout":        "miRNA holdout",
    "test_cell_holdout":         "cell holdout",
    "test_pair_x_cell_holdout":  "pair × cell holdout",
    "test_mirna_x_cell_holdout": "miRNA × cell holdout",
    "test_utr_holdout":          "UTR holdout",
    "test_utr_x_cell_holdout":   "UTR × cell holdout",
}

# Lymphoid first, then myeloid (display order)
CELL_ORDER_HINT = [
    "CD8n", "CD8a", "Tdg", "CD4", "NK2", "NK1", "B2",
    "B1", "Tregs", "plasma",
    "pDC", "mDC1", "mDC2", "Mono", "ncMono", "Gran",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report", required=True, help="path to final_report.json")
    p.add_argument("--out_dir", default=str(DIAG_DIR))
    p.add_argument("--out_name", default="fig43_per_cell_heatmap")
    p.add_argument("--title", default=None)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rep = json.loads(Path(args.report).read_text())
    evs = rep.get("evaluations", {})

    # Collect cell types
    cells_seen = set()
    for ts in TEST_ORDER:
        for pc in evs.get(ts, {}).get("per_cell", []):
            cells_seen.add(pc["cell_type"])
    # Order cells: known hint first, unknown alphabetically last
    cells = [c for c in CELL_ORDER_HINT if c in cells_seen]
    cells += sorted([c for c in cells_seen if c not in CELL_ORDER_HINT])

    if not cells:
        print("[error] no per_cell breakdown found in report")
        return

    # Build matrix
    M = np.full((len(TEST_ORDER), len(cells)), np.nan)
    for i, ts in enumerate(TEST_ORDER):
        ev = evs.get(ts, {})
        for pc in ev.get("per_cell", []):
            ct = pc.get("cell_type")
            if ct not in cells:
                continue
            j = cells.index(ct)
            M[i, j] = pc.get("pr_auc", np.nan)

    # ---- Plot ----
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
    })

    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(cells) + 3), 0.55 * len(TEST_ORDER) + 2))
    # Diverging norm centered at ~random baseline (0.30) so values around the
    # baseline appear near-white, well-above appear blue, below appear yellow.
    norm = TwoSlopeNorm(vmin=0.15, vcenter=0.30, vmax=1.0)
    im = ax.imshow(M, aspect="auto", cmap=CMAP_PR, norm=norm)

    ax.set_xticks(np.arange(len(cells)))
    ax.set_xticklabels(cells, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(TEST_ORDER)))
    ax.set_yticklabels([TEST_LABELS[t] for t in TEST_ORDER], fontsize=9)

    # Annotate cells with value (white text on saturated yellow/blue, dark elsewhere)
    for i in range(len(TEST_ORDER)):
        for j in range(len(cells)):
            v = M[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=7, color="#777")
            else:
                color = "white" if (v > 0.65 or v < 0.20) else "#222"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("PR-AUC", fontsize=10)

    title = args.title or f"Per-cell PR-AUC — {rep.get('config', {}).get('run_tag', '')}"
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=8)

    fig.tight_layout()
    out_png = out_dir / f"{args.out_name}.png"
    out_pdf = out_dir / f"{args.out_name}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"[save] {out_png}")
    print(f"[save] {out_pdf}")


if __name__ == "__main__":
    main()
