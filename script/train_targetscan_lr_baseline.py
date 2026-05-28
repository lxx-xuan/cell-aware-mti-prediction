#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TargetScan-only logistic regression baseline.

Trains a sklearn LogisticRegression on the 5-d TargetScan hand-crafted
features alone (no neural network, no cell-conditioning), evaluates on all
held-out splits, and writes a `final_report.json` in the same format as
the V3/V4 training scripts — so that downstream plotting (Fig 4.2) can
read it like any other model run.

This baseline represents "what you can do with classical hand-crafted
miRNA-target features alone" — i.e. the strongest fair cross-method
comparison given the cell-aware task setup.

Reads:
    cache/targetscan_features.npz   (pair_mirna, pair_utr, features)
    cache/splits/{train,val,test_*}_rows.npz

Writes:
    runs/targetscan_lr_baseline/final_report.json

Usage:
    python script/train_targetscan_lr_baseline.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score


SCRIPT_DIR = Path(__file__).resolve().parent
V3_ROOT = SCRIPT_DIR.parent
CACHE_DIR = V3_ROOT / "cache"

TEST_SPLITS_ORDER = [
    "val",
    "test_triple",
    "test_mirna_holdout",
    "test_cell_holdout",
    "test_pair_x_cell_holdout",
    "test_mirna_x_cell_holdout",
    "test_utr_holdout",
    "test_utr_x_cell_holdout",
]


def load_ts_features(cache_dir: Path):
    """Return dict {(mirna_gid, utr_gid): 5d feat} and full feature matrix."""
    p = cache_dir / "targetscan_features.npz"
    if not p.exists():
        print(f"[error] {p} not found", file=sys.stderr)
        sys.exit(1)
    d = np.load(p)
    pm = d["pair_mirna"].astype(int)
    pu = d["pair_utr"].astype(int)
    pf = d["features"].astype(np.float32)
    feat_dict = {(int(pm[i]), int(pu[i])): pf[i] for i in range(len(pm))}
    return feat_dict, pf


def build_X_y_split(split_path: Path, ts_dict: dict, default_feat: np.ndarray):
    d = np.load(split_path, allow_pickle=True)
    mirna = d["mirna_gid"].astype(int)
    utr = d["utr_gid"].astype(int)
    y = d["label"].astype(int)
    cell_type = d["cell_type"].astype(str) if "cell_type" in d.files else None
    X = np.zeros((len(y), default_feat.shape[0]), dtype=np.float32)
    miss = 0
    for i, (m, u) in enumerate(zip(mirna, utr)):
        feat = ts_dict.get((int(m), int(u)))
        if feat is None:
            X[i] = default_feat
            miss += 1
        else:
            X[i] = feat
    return X, y, cell_type, miss


def eval_split(X, y, clf, scaler, cell_type=None):
    X_s = scaler.transform(X)
    prob = clf.predict_proba(X_s)[:, 1]
    try:
        pr = float(average_precision_score(y, prob))
    except Exception:
        pr = float("nan")
    try:
        roc = float(roc_auc_score(y, prob))
    except Exception:
        roc = float("nan")
    out = {
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "pr_auc": pr,
        "roc_auc": roc,
    }
    if cell_type is not None:
        per_cell = []
        ct = np.asarray(cell_type).astype(str)
        for c in sorted(set(ct)):
            mask = ct == c
            if mask.sum() < 100:
                continue
            try:
                pra = float(average_precision_score(y[mask], prob[mask]))
                roa = float(roc_auc_score(y[mask], prob[mask]))
            except Exception:
                pra, roa = float("nan"), float("nan")
            per_cell.append({
                "cell_type": str(c),
                "n": int(mask.sum()),
                "positive_rate": float(y[mask].mean()),
                "pr_auc": pra,
                "roc_auc": roa,
            })
        out["per_cell"] = per_cell
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", default=str(CACHE_DIR))
    p.add_argument("--out_dir", default=str(V3_ROOT / "runs" / "targetscan_lr_baseline"))
    p.add_argument("--C", type=float, default=1.0,
                   help="L2 regularization strength (sklearn convention; default 1.0)")
    p.add_argument("--max_iter", type=int, default=2000)
    args = p.parse_args()

    cache = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[load] TS features from {cache / 'targetscan_features.npz'}")
    ts_dict, ts_all = load_ts_features(cache)
    print(f"  unique (miRNA, UTR) pairs: {len(ts_dict):,}")
    print(f"  feature dim: {ts_all.shape[1]}")

    # Use mean of all TS features as default for missing pairs
    default_feat = ts_all.mean(axis=0)

    train_path = cache / "splits" / "train_rows.npz"
    print(f"[load] train: {train_path}")
    X_tr, y_tr, _, miss_tr = build_X_y_split(train_path, ts_dict, default_feat)
    print(f"  n={len(y_tr):,}  pos_rate={y_tr.mean():.4f}  "
          f"missing_features={miss_tr:,} ({miss_tr/len(y_tr):.2%})")

    print(f"[train] StandardScaler + LogisticRegression(C={args.C}, max_iter={args.max_iter})")
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    clf = LogisticRegression(C=args.C, max_iter=args.max_iter,
                              solver="lbfgs", n_jobs=-1)
    clf.fit(X_tr_s, y_tr)
    print(f"  feature coef:  {np.round(clf.coef_[0], 4).tolist()}")
    print(f"  intercept:     {float(clf.intercept_[0]):.4f}")
    print(f"  train PR-AUC:  "
          f"{average_precision_score(y_tr, clf.predict_proba(X_tr_s)[:, 1]):.4f}")

    # ---- Evaluate ----
    results = {
        "config": {
            "run_tag": "targetscan_lr_baseline",
            "model_kind": "targetscan_lr",
            "feature_dim": int(ts_all.shape[1]),
            "classifier": f"LogisticRegression(C={args.C}, solver=lbfgs)",
            "scaler": "StandardScaler",
        },
        "best_epoch": 1,
        "evaluations": {},
    }
    print()
    print(f"{'split':>30}  {'n':>10}  {'pos_rate':>10}  {'PR-AUC':>8}  {'ROC-AUC':>8}")
    for split in TEST_SPLITS_ORDER:
        sp = cache / "splits" / f"{split}_rows.npz"
        if not sp.exists():
            continue
        X, y, ct, miss = build_X_y_split(sp, ts_dict, default_feat)
        ev = eval_split(X, y, clf, scaler, ct)
        results["evaluations"][split] = ev
        print(f"  {split:>28}  {ev['n']:>10,}  {ev['positive_rate']:>10.4f}  "
              f"{ev['pr_auc']:>8.4f}  {ev['roc_auc']:>8.4f}")

    out_json = out_dir / "final_report.json"
    with out_json.open("w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n[save] {out_json}")
    print(f"[done] elapsed={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
