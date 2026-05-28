#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig 4.2 (b) — grouped bar chart comparing all methods on 7 test sets.

Reads `final_report.json` from each run folder (TargetScan-LR + V3-full +
V3-mixed + V4-full + V4-mixed) and produces a journal-style grouped bar
chart, with a dashed random-baseline reference line.

Outputs:
    runs/_diagnostics/fig42_model_comparison.png
    runs/_diagnostics/fig42_model_comparison.pdf

Usage:
    python script/plot_fig42_comparison.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


SCRIPT_DIR = Path(__file__).resolve().parent
V3_ROOT = SCRIPT_DIR.parent
DIAG_DIR = V3_ROOT / "runs" / "_diagnostics"

# Test sets ordered left-to-right: fewer axes held out → more axes held out
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
    "test_triple":              "triple\n(all seen)",
    "test_mirna_holdout":       "miRNA\nholdout",
    "test_cell_holdout":        "cell\nholdout",
    "test_pair_x_cell_holdout": "pair×cell\nholdout",
    "test_mirna_x_cell_holdout": "miRNA×cell\nholdout",
    "test_utr_holdout":         "UTR\nholdout",
    "test_utr_x_cell_holdout":  "UTR×cell\nholdout",
}

# (label, run-folder name)
# All labels are ASCII to avoid CJK font dependency; the figure caption
# maps these back to the Chinese model names used in the paper body.
DEFAULT_RUNS = [
    ("TargetScan-LR",                  "targetscan_lr_baseline"),
    ("CNN baseline (full)",            "main_film_seed42"),
    ("CNN baseline (mixed-only)",      "diag_mixed_film_seed42"),
    ("ConvNeXt extended (full)",       "v4_film_seed42"),
    ("ConvNeXt extended (mixed-only)", "v4_film_mixed_seed42"),
]

# Unified palette anchored on #5B9BD5 (lymphoid blue) + #F4C430 (myeloid yellow)
COLORS = {
    "TargetScan-LR":                  "#999999",   # neutral grey baseline
    "CNN baseline (full)":            "#5B9BD5",   # anchor blue (main model)
    "CNN baseline (mixed-only)":      "#B5D3EC",   # lighter blue (diagnostic)
    "ConvNeXt extended (full)":       "#F4C430",   # anchor yellow (extended)
    "ConvNeXt extended (mixed-only)": "#FAE39B",   # lighter yellow (diagnostic)
}


def load_run(path: Path):
    if not path.exists():
        return None
    with path.open() as fh:
        return json.load(fh)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default=str(V3_ROOT / "runs"))
    p.add_argument("--out_dir", default=str(DIAG_DIR))
    p.add_argument("--runs", nargs="+", default=None,
                   help="optional override: 'label:run_folder' pairs")
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.runs:
        runs = [tuple(r.split(":", 1)) for r in args.runs]
    else:
        runs = DEFAULT_RUNS

    # ---- Load ----
    data = {}      # {label: {test_set: pr_auc}}
    pos_rates = {} # {test_set: positive_rate}
    for label, folder in runs:
        rep_path = runs_dir / folder / "final_report.json"
        rep = load_run(rep_path)
        if rep is None:
            print(f"[skip] {label}: {rep_path} not found")
            continue
        evs = rep.get("evaluations", {})
        data[label] = {}
        for ts in TEST_ORDER:
            ev = evs.get(ts)
            if ev is None:
                continue
            pr = ev.get("pr_auc")
            data[label][ts] = pr
            if ts not in pos_rates and "positive_rate" in ev:
                pos_rates[ts] = float(ev["positive_rate"])
        print(f"[load] {label:>16}  ({len(data[label])} test sets)")

    if not data:
        print("[error] no runs loaded", file=sys.stderr)
        sys.exit(1)

    # ---- Plot ----
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
    })

    fig, ax = plt.subplots(figsize=(13, 5))
    n_methods = len(data)
    n_tests = len(TEST_ORDER)
    bar_width = 0.78 / n_methods
    x = np.arange(n_tests)

    for i, (label, vals) in enumerate(data.items()):
        ys = [vals.get(ts, np.nan) for ts in TEST_ORDER]
        offset = (i - (n_methods - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset, ys, width=bar_width, label=label,
            color=COLORS.get(label, f"C{i}"),
            edgecolor="#333333", linewidth=0.4,
        )
        # value labels
        for bar, v in zip(bars, ys):
            if not np.isnan(v) and v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        v + 0.012, f"{v:.2f}",
                        ha="center", fontsize=6.5, color="#333", rotation=0)

    # Baseline reference line
    if pos_rates:
        avg_base = float(np.mean(list(pos_rates.values())))
        ax.axhline(avg_base, color="#444", linestyle="--", linewidth=0.9,
                    label=f"Random baseline ≈ {avg_base:.2f}", zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels([TEST_LABELS[ts] for ts in TEST_ORDER], fontsize=9)
    ax.set_ylabel("PR-AUC", fontsize=11)
    ax.set_xlabel("Test set (more axes held out →)", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color="#D0D0D0", alpha=0.6, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, loc="upper right", frameon=False, ncol=2)

    fig.tight_layout()
    out_png = out_dir / "fig42_model_comparison.png"
    out_pdf = out_dir / "fig42_model_comparison.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"\n[save] {out_png}")
    print(f"[save] {out_pdf}")


if __name__ == "__main__":
    main()
