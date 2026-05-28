#!/usr/bin/env bash
# ============================================================================
# V3 — DIAGNOSTIC: train on mixed pairs only.
#
# Runs train_main_model_mixed_only.py for ONE variant (default: film).
# Re-uses the cache already built by the regular run_all.sh (does NOT rebuild).
#
# Why: see docstring at the top of train_main_model_mixed_only.py.
# In short: forces the model to use cell information by removing pure pairs
# from training. Lets us tell whether the test_utr_holdout collapse is caused
# by (a) UTR-identity shortcut in training, or (b) the conv branches' inability
# to learn transferable seed-level features. If (a), this run should improve
# test_utr_holdout. If (b), it won't.
#
# Usage:
#   ./script/run_mixed_only.sh                       # film, default knobs
#   MODEL=full ./script/run_mixed_only.sh            # try the 'full' variant
#   MAX_EPOCHS=15 ./script/run_mixed_only.sh         # train longer
#   SEED=7 ./script/run_mixed_only.sh                # different seed
#
# Tag (default 'diag_mixed_<MODEL>_seed<SEED>') is picked so that
# `python script/summarize_results.py --include_tags diag_mixed_ main_`
# will produce a side-by-side comparison with the main run.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V3_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${V3_ROOT}"
echo "[mixed-only] V3_ROOT=${V3_ROOT}"

# --- tunables (mirror run_all.sh defaults so the cache matches) ---
MODEL="${MODEL:-film}"
HOLDOUT="${HOLDOUT:-B1,Tregs,plasma,mDC1}"
VAL_HOLDOUT_CELLS="${VAL_HOLDOUT_CELLS:-mDC1}"
LOSS_KIND="${LOSS_KIND:-focal_balanced}"
FOCAL_GAMMA="${FOCAL_GAMMA:-2.0}"
MAX_EPOCHS="${MAX_EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-42}"
LR="${LR:-1e-3}"
WD="${WD:-1e-4}"
CELL_EMBED_DIM="${CELL_EMBED_DIM:-32}"
PATIENCE="${PATIENCE:-3}"
MAX_CHUNKS="${MAX_CHUNKS:-0}"
LOG_EVERY="${LOG_EVERY:-200}"
SEED_RULE="${SEED_RULE:-any_7mer}"
UTR_HOLDOUT_FRAC="${UTR_HOLDOUT_FRAC:-0.20}"
MIRNA_HOLDOUT_FRAC="${MIRNA_HOLDOUT_FRAC:-0.20}"
TAG="${TAG:-diag_mixed_${MODEL}_seed${SEED}}"

echo "[mixed-only] model=${MODEL}  tag=${TAG}"
echo "[mixed-only] holdout=${HOLDOUT}  val_holdout=${VAL_HOLDOUT_CELLS}"
echo "[mixed-only] loss=${LOSS_KIND} (gamma=${FOCAL_GAMMA})  max_epochs=${MAX_EPOCHS}  seed=${SEED}"

mkdir -p runs

# Sanity: cache must already exist. We do NOT rebuild it.
if [[ ! -f "cache/splits/train_rows.npz" ]]; then
    echo "[error] cache/splits/train_rows.npz not found."
    echo "[error] Please run ./script/run_all.sh first to build the cache."
    exit 1
fi
echo "[mixed-only] cache present; will reuse it"

# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "[diag] TRAINING variant=${MODEL} on MIXED PAIRS ONLY  ->  run_tag=${TAG}"
echo "============================================================"
python script/train_main_model_mixed_only.py \
    --run_tag "${TAG}" \
    --model "${MODEL}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --device "${DEVICE}" \
    --max_epochs "${MAX_EPOCHS}" \
    --patience "${PATIENCE}" \
    --lr "${LR}" \
    --weight_decay "${WD}" \
    --cell_embed_dim "${CELL_EMBED_DIM}" \
    --seed "${SEED}" \
    --seed_rule "${SEED_RULE}" \
    --utr_holdout_frac "${UTR_HOLDOUT_FRAC}" \
    --mirna_holdout_frac "${MIRNA_HOLDOUT_FRAC}" \
    --val_holdout_cell_types "${VAL_HOLDOUT_CELLS}" \
    --loss_kind "${LOSS_KIND}" \
    --focal_gamma "${FOCAL_GAMMA}" \
    --holdout_cell_types "${HOLDOUT}" \
    --log_every_batches "${LOG_EVERY}" \
    --max_chunks "${MAX_CHUNKS}"

echo ""
echo "============================================================"
echo "[diag] DONE"
echo "============================================================"
echo "[mixed-only] report: ${V3_ROOT}/runs/${TAG}/final_report.json"
echo "[mixed-only] compare with main run:"
echo "    python script/summarize_results.py --include_tags ${TAG} main_${MODEL}"
