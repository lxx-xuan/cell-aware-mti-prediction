#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Leakage check across all 5 evaluation sets.

For every test split, reports:
  1. unique (mirna, utr) pairs and how many were already in train
  2. row-level pair leakage and (pair, label) leakage
  3. unique miRNAs / UTRs seen in train

Then asserts the invariants that MUST hold for a paper-defensible split:
    train ∩ val           pair overlap == 0
    train ∩ test_triple   pair overlap == 0
    train ∩ test_pair_x_cell_holdout pair overlap == 0
(test_cell_holdout intentionally MAY overlap with train at the pair level —
 it's the "new cell, possibly seen pair" set, kept for contrast.)
"""

from pathlib import Path
import sys
import numpy as np

CACHE = Path(__file__).resolve().parent.parent / "cache" / "splits"

ALL_SPLITS = [
    "train", "val", "test_triple",
    "val_cell_holdout", "test_cell_holdout", "test_pair_x_cell_holdout",
    "val_utr_holdout", "val_utr_x_cell_holdout",
    "test_utr_holdout", "test_utr_x_cell_holdout",
    "test_mirna_holdout", "test_mirna_x_cell_holdout",
]


def load(name: str):
    p = CACHE / f"{name}_rows.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}


def pair_set(d):
    return set(zip(d["mirna_gid"].tolist(), d["utr_gid"].tolist()))


def pair_label_dict(d):
    out = {}
    for m, u, y in zip(d["mirna_gid"].tolist(), d["utr_gid"].tolist(), d["label"].tolist()):
        out.setdefault((int(m), int(u)), set()).add(int(y))
    return out


sets = {n: load(n) for n in ALL_SPLITS}
missing = [n for n, d in sets.items() if d is None]
if missing:
    print(f"[warn] missing splits: {missing}")

print(f"{'split':<28} {'n_rows':>10} {'n_unique_pairs':>16} {'n_unique_mirna':>16} {'n_unique_utr':>14}")
for name in ALL_SPLITS:
    d = sets[name]
    if d is None:
        continue
    pairs = pair_set(d)
    mirnas = set(d["mirna_gid"].tolist())
    utrs = set(d["utr_gid"].tolist())
    print(f"{name:<28} {len(d['label']):>10,} {len(pairs):>16,} {len(mirnas):>16,} {len(utrs):>14,}")

if sets["train"] is None:
    print("[error] no train split; aborting")
    sys.exit(1)

train = sets["train"]
train_pairs = pair_set(train)
train_pair_labels = pair_label_dict(train)
train_mirnas = set(train["mirna_gid"].tolist())
train_utrs = set(train["utr_gid"].tolist())

# splits that MUST be pair-disjoint from train (pair-level holdout sets)
strict_clean_pair = ["val", "test_triple", "test_pair_x_cell_holdout"]
# splits that MUST be UTR-disjoint from train (UTR holdout sets)
strict_clean_utr = ["val_utr_holdout", "val_utr_x_cell_holdout",
                    "test_utr_holdout", "test_utr_x_cell_holdout"]
# splits that MUST be miRNA-disjoint from train (miRNA holdout sets)
strict_clean_mirna = ["test_mirna_holdout", "test_mirna_x_cell_holdout"]
# splits that may have pair overlap with train (intentional contrast)
intentional_leak = ["test_cell_holdout", "val_cell_holdout"]

print()
violations = []
for name in [n for n in ALL_SPLITS if n != "train" and sets[n] is not None]:
    d = sets[name]
    pairs = pair_set(d)
    n_pair = len(pairs)
    n_pair_seen = len(pairs & train_pairs)
    rows_pair_seen = 0
    rows_pair_same_label = 0
    for m, u, y in zip(d["mirna_gid"].tolist(), d["utr_gid"].tolist(), d["label"].tolist()):
        if (m, u) in train_pair_labels:
            rows_pair_seen += 1
            if int(y) in train_pair_labels[(m, u)]:
                rows_pair_same_label += 1
    n_mirna = len(set(d["mirna_gid"].tolist()))
    n_utr = len(set(d["utr_gid"].tolist()))
    n_mirna_seen = len(set(d["mirna_gid"].tolist()) & train_mirnas)
    n_utr_seen = len(set(d["utr_gid"].tolist()) & train_utrs)

    tags = []
    if name in strict_clean_pair:
        tags.append("STRICT-CLEAN-PAIR")
    if name in strict_clean_utr:
        tags.append("STRICT-CLEAN-UTR")
    if name in strict_clean_mirna:
        tags.append("STRICT-CLEAN-MIRNA")
    if name in intentional_leak:
        tags.append("LEAK-ALLOWED")
    tag = " ".join(f"[{t}]" for t in tags) if tags else ""
    print(f"=== {name}  {tag} ===")
    print(f"  unique (mirna, utr) pairs:   {n_pair:>7,}")
    print(f"    of which already in train: {n_pair_seen:>7,}  "
          f"({n_pair_seen / max(n_pair, 1):.1%})")
    print(f"  rows whose (mirna, utr) was in train:        {rows_pair_seen:>7,} / {len(d['label']):>7,}  "
          f"({rows_pair_seen / max(len(d['label']), 1):.1%})")
    print(f"  rows whose (mirna, utr, label) was in train: {rows_pair_same_label:>7,} / {len(d['label']):>7,}  "
          f"({rows_pair_same_label / max(len(d['label']), 1):.1%})")
    print(f"  unique miRNAs:  {n_mirna:>4}  of which seen in train: {n_mirna_seen:>4}  "
          f"({n_mirna_seen / max(n_mirna, 1):.1%})")
    print(f"  unique UTRs:    {n_utr:>5,}  of which seen in train: {n_utr_seen:>5,}  "
          f"({n_utr_seen / max(n_utr, 1):.1%})")
    print()

    # Strict invariants:
    if name in strict_clean_pair and n_pair_seen > 0:
        violations.append((name, "PAIR", n_pair_seen, rows_pair_seen))
    if name in strict_clean_utr and n_utr_seen > 0:
        violations.append((name, "UTR", n_utr_seen, 0))
    if name in strict_clean_mirna and n_mirna_seen > 0:
        violations.append((name, "MIRNA", n_mirna_seen, 0))

# test_pair_x_cell_holdout must be a SUBSET of test_cell_holdout
if sets.get("test_pair_x_cell_holdout") is not None and sets.get("test_cell_holdout") is not None:
    p_pxch = pair_set(sets["test_pair_x_cell_holdout"])
    p_ch = pair_set(sets["test_cell_holdout"])
    if not p_pxch.issubset(p_ch):
        print(f"[WARN] test_pair_x_cell_holdout pairs not a subset of test_cell_holdout pairs "
              f"(diff={len(p_pxch - p_ch)})")
    else:
        print("[ok] test_pair_x_cell_holdout pairs ⊆ test_cell_holdout pairs")

if violations:
    print("\n!!! LEAKAGE VIOLATION !!!")
    for v in violations:
        name, axis, n_overlap, n_rows = v
        print(f"  {name} [{axis}]: {n_overlap:,} train-overlapping {axis.lower()} values"
              + (f", {n_rows:,} rows" if n_rows else ""))
    sys.exit(2)
else:
    print("\n=== LEAKAGE CHECK PASSED ===")
