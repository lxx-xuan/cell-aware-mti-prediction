#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scan all runs/<tag>/final_report.json and emit a comparison table.

Outputs (under runs/_summary/):
    main_table.csv       one row per (run_tag, split), columns = key metrics
    main_table.md        same content rendered as a markdown table
    per_cell_table.csv   one row per (run_tag, split, cell_type)
    per_cell_table.md
    summary_log.txt      pretty-printed copy of main_table.md (for grep / paste)

Usage:
    python script/summarize_results.py
    python script/summarize_results.py --runs_dir /path/to/runs
    python script/summarize_results.py --include_tags full no_cell no_attn cell_only
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
V3_ROOT = SCRIPT_DIR.parent
RUNS_DIR_DEFAULT = V3_ROOT / "runs"

# Splits to report (in display order)
SPLIT_ORDER = [
    "val",
    "val_utr_holdout", "val_cell_holdout", "val_utr_x_cell_holdout",
    "test_triple",
    "test_cell_holdout", "test_pair_x_cell_holdout",
    "test_utr_holdout", "test_utr_x_cell_holdout",
    "test_mirna_holdout", "test_mirna_x_cell_holdout",
]
# Columns to report (in display order)
METRIC_COLS = [
    ("n", "n"),
    ("positive_rate", "pos_rate"),
    ("pr_auc", "PR-AUC"),
    ("roc_auc", "ROC-AUC"),
    ("best_f1", "F1"),
    ("best_threshold", "thr"),
    ("p_at_top_1pct", "P@1%"),
    ("p_at_top_5pct", "P@5%"),
    ("p_at_top_10pct", "P@10%"),
]


def discover_runs(runs_dir: Path) -> list[tuple[str, dict]]:
    found = []
    for sub in sorted(runs_dir.iterdir()) if runs_dir.exists() else []:
        if not sub.is_dir():
            continue
        fr = sub / "final_report.json"
        if not fr.exists():
            continue
        try:
            data = json.loads(fr.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[warn] cannot read {fr}: {e}")
            continue
        found.append((sub.name, data))
    return found


def fmt(v, key):
    if v is None:
        return ""
    if key == "n":
        return f"{int(v):,}"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def render_markdown(rows: list[list[str]], header: list[str]) -> str:
    widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(header)]
    def line(cells):
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    out = [line(header), sep]
    out += [line(r) for r in rows]
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default=str(RUNS_DIR_DEFAULT))
    p.add_argument("--include_tags", nargs="+", default=None,
                   help="if given, only include these run_tags (substring match)")
    p.add_argument("--out_dir", default=None,
                   help="output dir (default: <runs_dir>/_summary)")
    a = p.parse_args()
    runs_dir = Path(a.runs_dir)
    out_dir = Path(a.out_dir) if a.out_dir else runs_dir / "_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(runs_dir)
    if a.include_tags:
        runs = [(t, d) for t, d in runs if any(s in t for s in a.include_tags)]
    print(f"found {len(runs)} runs in {runs_dir}")
    for tag, _ in runs:
        print(f"  - {tag}")

    # ---- main table ----
    header = ["run_tag", "split", "model_kind", "best_epoch"] + [c[1] for c in METRIC_COLS]
    rows: list[list[str]] = []
    for tag, data in runs:
        model_kind = (data.get("config") or {}).get("model_kind", "?")
        best_ep = data.get("best_epoch", "?")
        evals = data.get("evaluations", {})
        for split in SPLIT_ORDER:
            ev = evals.get(split)
            if ev is None:
                continue
            row = [tag, split, str(model_kind), str(best_ep)]
            for k, _ in METRIC_COLS:
                row.append(fmt(ev.get(k), k))
            rows.append(row)

    main_csv = out_dir / "main_table.csv"
    with main_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    main_md = render_markdown(rows, header)
    (out_dir / "main_table.md").write_text(main_md + "\n", encoding="utf-8")
    print(f"\n=== MAIN TABLE ===\n{main_md}\n")

    # ---- per-cell table ----
    pc_header = ["run_tag", "split", "cell_type", "n", "pos_rate", "PR-AUC", "ROC-AUC"]
    pc_rows: list[list[str]] = []
    for tag, data in runs:
        evals = data.get("evaluations", {})
        for split in SPLIT_ORDER:
            ev = evals.get(split)
            if ev is None:
                continue
            for cell_row in ev.get("per_cell", []) or []:
                pc_rows.append([
                    tag, split,
                    cell_row.get("cell_type", "?"),
                    f"{int(cell_row.get('n', 0)):,}",
                    f"{cell_row.get('positive_rate', 0.0):.4f}",
                    f"{cell_row.get('pr_auc', 0.0):.4f}",
                    f"{cell_row.get('roc_auc', 0.0):.4f}",
                ])
    with (out_dir / "per_cell_table.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(pc_header)
        w.writerows(pc_rows)
    pc_md = render_markdown(pc_rows, pc_header)
    (out_dir / "per_cell_table.md").write_text(pc_md + "\n", encoding="utf-8")

    (out_dir / "summary_log.txt").write_text(
        "=== MAIN TABLE ===\n" + main_md + "\n\n=== PER-CELL TABLE ===\n" + pc_md + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out_dir / 'main_table.md'}")
    print(f"wrote {out_dir / 'main_table.csv'}")
    print(f"wrote {out_dir / 'per_cell_table.md'}")
    print(f"wrote {out_dir / 'per_cell_table.csv'}")


if __name__ == "__main__":
    main()
