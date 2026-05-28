#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare seed-target matching rules WITHOUT retraining.

Reads cache/global_maps/{mirna_seqs.json, utr_seqs.pkl} and cache/splits/*.npz,
then for every (mirna_gid, utr_gid) pair across all splits computes whether each
rule's seed target(s) can be found in the UTR. Reports hit rate per rule.

Use this to decide whether to switch the default seed rule before tomorrow's
big Ubuntu run.

Rules (TargetScan naming):
  7mer-m8:  RC(miRNA[2..8])                            -- CURRENT DEFAULT
  7mer-A1:  RC(miRNA[2..7]) + "A"
  6mer:     RC(miRNA[2..7])
  8mer:     RC(miRNA[2..8]) + "A"
  any_7mer: 7mer-m8 OR 7mer-A1   (hit if EITHER matches)
  any:      8mer OR 7mer-m8 OR 7mer-A1 OR 6mer
"""

from pathlib import Path
import json
import pickle
import numpy as np

V3 = Path(__file__).resolve().parent.parent
CACHE = V3 / "cache"

RC_DNA = str.maketrans("ACGTUacgtu", "TGCAATGCAA")


def rc(s: str) -> str:
    return s.translate(RC_DNA)[::-1]


def targets_for(mseq: str, rule: str) -> list[str]:
    seg_2_7 = mseq[1:7]   # miRNA positions 2..7 (1-indexed), 6 chars
    seg_2_8 = mseq[1:8]   # miRNA positions 2..8, 7 chars
    if rule == "7mer-m8":
        return [rc(seg_2_8)]                  # 7 chars
    if rule == "7mer-A1":
        return [rc(seg_2_7) + "A"]             # 7 chars
    if rule == "6mer":
        return [rc(seg_2_7)]                   # 6 chars
    if rule == "8mer":
        return [rc(seg_2_8) + "A"]             # 8 chars
    if rule == "any_7mer":
        return [rc(seg_2_8), rc(seg_2_7) + "A"]
    if rule == "any":
        return [rc(seg_2_8) + "A", rc(seg_2_8), rc(seg_2_7) + "A", rc(seg_2_7)]
    raise ValueError(rule)


def normalize(s: str) -> str:
    return s.upper().replace("U", "T")


def hit_rate(pairs: list[tuple[int, int]], mseqs: list[str], useqs: list[str], rule: str):
    if not pairs:
        return 0.0, 0, 0
    n_hit = 0
    for mid, uid in pairs:
        targets = [normalize(t) for t in targets_for(mseqs[mid], rule)]
        if not all(targets):
            continue
        utr_norm = normalize(useqs[uid])
        if any(t in utr_norm for t in targets):
            n_hit += 1
    return n_hit / len(pairs), n_hit, len(pairs)


def main():
    mseqs = json.loads((CACHE / "global_maps" / "mirna_seqs.json").read_text(encoding="utf-8"))
    with open(CACHE / "global_maps" / "utr_seqs.pkl", "rb") as fh:
        useqs = pickle.load(fh)
    print(f"loaded {len(mseqs)} mirnas, {len(useqs)} utrs")

    # union of pairs across all splits
    all_pairs: set[tuple[int, int]] = set()
    per_split: dict[str, list[tuple[int, int]]] = {}
    for name in ["train", "val", "test_triple", "test_cell_holdout", "test_pair_x_cell_holdout"]:
        p = CACHE / "splits" / f"{name}_rows.npz"
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=True)
        pairs = list(set(zip(z["mirna_gid"].tolist(), z["utr_gid"].tolist())))
        per_split[name] = pairs
        all_pairs.update(pairs)
    all_pairs_list = list(all_pairs)
    print(f"union unique pairs across all splits: {len(all_pairs_list):,}")
    print()

    rules = ["7mer-m8", "7mer-A1", "6mer", "8mer", "any_7mer", "any"]

    # union-level table
    print(f"{'rule':<10} {'seed_target_len':<16} {'hit_rate':<10} {'n_hit / n_pairs':<20}")
    print("-" * 60)
    for rule in rules:
        seed_lens = sorted(set(len(t) for t in targets_for(mseqs[0], rule)))
        hr, nh, n = hit_rate(all_pairs_list, mseqs, useqs, rule)
        marker = "  <-- CURRENT" if rule == "7mer-m8" else ""
        print(f"{rule:<10} {str(seed_lens):<16} {hr:>8.4f}  {nh:>6,} / {n:<8,}{marker}")
    print()

    # per-split table for the "any_7mer" rule (good middle ground)
    print(f"per-split hit rate for rule='any_7mer':")
    for name, pairs in per_split.items():
        hr, nh, n = hit_rate(pairs, mseqs, useqs, "any_7mer")
        print(f"  {name:<28} {hr:>7.4f}  ({nh:,} / {n:,})")


if __name__ == "__main__":
    main()
