#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot the rank-aware new-label threshold sweep heatmap (Methods §3.x sensitivity
analysis figure) — sample size vs class balance trade-off.

Data is hardcoded from the original sensitivity analysis (recovered from a
previously-rendered figure) so the script is self-contained. To re-color or
re-style, edit the CONFIG block at the top.

Outputs:
    runs/_diagnostics/fig_threshold_sweep.png
    runs/_diagnostics/fig_threshold_sweep.pdf

Usage:
    python script/plot_threshold_sweep.py
    python script/plot_threshold_sweep.py --cmap Blues
    python script/plot_threshold_sweep.py --cmap viridis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# =========================== CONFIG (edit here) ===========================
# Unified palette anchored on #5B9BD5 (lymphoid blue) and #F4C430 (myeloid yellow)
ANCHOR_BLUE   = "#5B9BD5"
ANCHOR_YELLOW = "#F4C430"
DEEP_BLUE     = "#1F4E79"

# Sequential white → blue → deep blue (for sample-count-like quantities)
CMAP_SEQ_BLUE = LinearSegmentedColormap.from_list(
    "cmap_seq_blue", ["#FFFFFF", "#D6E6F4", ANCHOR_BLUE, DEEP_BLUE])

# Diverging yellow → white → blue (for ratio-like quantities; midpoint = neutral)
CMAP_PR = LinearSegmentedColormap.from_list(
    "cmap_pr", [ANCHOR_YELLOW, "#FFFFFF", ANCHOR_BLUE, DEEP_BLUE])

DEFAULT_CMAP_LEFT  = CMAP_SEQ_BLUE   # Retained samples panel
DEFAULT_CMAP_RIGHT = CMAP_PR         # Positive rate panel
SELECTED_NEG = 0.50
SELECTED_POS = 0.20
FIGSIZE = (12, 4.5)
ANNOT_FONTSIZE = 9
TITLE = "Rank-aware new-label threshold selection: sample size vs class balance"
# =========================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
V3_ROOT = SCRIPT_DIR.parent
OUT_DIR = V3_ROOT / "runs" / "_diagnostics"

# x = negative threshold (rank_percentile_low >)
NEG_THR = [0.40, 0.50, 0.60, 0.70, 0.80]
# y = positive threshold (rank_percentile_low <), high → low (top to bottom)
POS_THR = [0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]

# Retained samples in millions (rows = POS_THR top→bottom, cols = NEG_THR left→right)
RETAINED_M = np.array([
    [46.9, 41.8, 36.0, 30.0, 24.3],
    [44.5, 39.4, 33.6, 27.6, 21.9],
    [42.2, 37.1, 31.3, 25.3, 19.6],
    [40.0, 34.8, 29.0, 23.1, 17.4],
    [37.8, 32.6, 26.9, 20.9, 15.2],
    [35.8, 30.6, 24.9, 18.9, 13.2],
    [34.1, 29.0, 23.2, 17.2, 11.5],
])

# Positive rate in % (same layout)
POSRATE_PCT = np.array([
    [30.0, 33.7, 39.2, 46.9, 58.0],
    [26.3, 29.7, 34.8, 42.3, 53.4],
    [22.3, 25.3, 30.0, 37.1, 47.9],
    [17.9, 20.3, 24.6, 31.0, 41.2],
    [13.2, 15.2, 18.5, 23.8, 32.8],
    [ 8.3,  9.7, 11.9, 15.7, 22.5],
    [ 3.9,  4.6,  5.7,  7.7, 11.5],
])


def _annotate(ax, M, fmt="{:.1f}"):
    nrow, ncol = M.shape
    vmin, vmax = M.min(), M.max()
    for i in range(nrow):
        for j in range(ncol):
            v = M[i, j]
            # white text on dark cells, black on light
            color = "white" if (v - vmin) / (vmax - vmin) > 0.55 else "#222"
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    fontsize=ANNOT_FONTSIZE, color=color)


def _mark_selected(ax, neg_val=SELECTED_NEG, pos_val=SELECTED_POS):
    j = NEG_THR.index(neg_val)
    i = POS_THR.index(pos_val)
    # red circle highlight
    ax.scatter([j], [i], s=600, facecolors="none",
               edgecolors="#D32F2F", linewidth=2.0, zorder=5)
    ax.text(j, i - 0.42, "selected", ha="center", va="bottom",
            fontsize=9, color="#D32F2F", fontweight="700")


def _setup_axes(ax, title):
    ax.set_xticks(range(len(NEG_THR)))
    ax.set_xticklabels([f"{v:.2f}" for v in NEG_THR], fontsize=10)
    ax.set_yticks(range(len(POS_THR)))
    ax.set_yticklabels([f"{v:.2f}" for v in POS_THR], fontsize=10)
    ax.set_xlabel("negative threshold:  rank_percentile_low  >", fontsize=10)
    ax.set_ylabel("positive threshold:  rank_percentile_low  <", fontsize=10)
    ax.set_title(title, fontsize=11, loc="left")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cmap_left", default=None,
                   help="colormap for the 'retained samples' panel (overrides unified palette)")
    p.add_argument("--cmap_right", default=None,
                   help="colormap for the 'positive rate' panel (overrides unified palette)")
    p.add_argument("--cmap", default=None,
                   help="shortcut: set both panels to the same colormap")
    p.add_argument("--out_dir", default=str(OUT_DIR))
    a = p.parse_args()

    cmap_left  = a.cmap or a.cmap_left  or DEFAULT_CMAP_LEFT
    cmap_right = a.cmap or a.cmap_right or DEFAULT_CMAP_RIGHT

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333",
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE,
                                    gridspec_kw=dict(wspace=0.35))

    # ---- left panel: retained samples ----
    im1 = ax1.imshow(RETAINED_M, cmap=cmap_left, aspect="auto")
    _annotate(ax1, RETAINED_M, fmt="{:.1f}")
    _setup_axes(ax1, "Retained samples under threshold sweep")
    _mark_selected(ax1)
    cb1 = plt.colorbar(im1, ax=ax1, shrink=0.85, pad=0.02)
    cb1.set_label("Rows retained (million)", fontsize=9)

    # ---- right panel: positive rate ----
    im2 = ax2.imshow(POSRATE_PCT, cmap=cmap_right, aspect="auto")
    _annotate(ax2, POSRATE_PCT, fmt="{:.1f}")
    _setup_axes(ax2, "Positive rate among retained samples")
    _mark_selected(ax2)
    cb2 = plt.colorbar(im2, ax=ax2, shrink=0.85, pad=0.02)
    cb2.set_label("Positive rate (%)", fontsize=9)

    fig.suptitle(TITLE, fontsize=12, y=1.02)
    fig.tight_layout()

    out_png = out_dir / "fig_threshold_sweep.png"
    out_pdf = out_dir / "fig_threshold_sweep.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"[save] {out_png}")
    print(f"[save] {out_pdf}")


if __name__ == "__main__":
    main()
