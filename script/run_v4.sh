#!/usr/bin/env bash
# ============================================================================
# V4 — ConvNeXt UTR encoder + 4 kb window + per-position seed-mask aux head.
#
# Reuses the cache built by run_all.sh (no rebuild needed).
# Default plan:
#   1.  V4-film  (main)           ~15-25 hours
#   2.  V4-film  (mixed-only)      ~3-5 hours    (set MIXED_ONLY=1)
#   3.  (optional ablations on success — comment-in below)
#
# Usage:
#   ./script/run_v4.sh                       # main: V4-film, full train
#   MIXED_ONLY=1 ./script/run_v4.sh          # diagnostic: train on mixed pairs only
#   MODEL=full ./script/run_v4.sh            # ablation: end-of-net concat instead of FiLM
#   NO_AUX=1 ./script/run_v4.sh              # ablation: disable aux seed head
#   NO_CONVNEXT=1 ./script/run_v4.sh         # ablation: fall back to V3 vanilla CNN
#   BATCH_SIZE=64 ./script/run_v4.sh         # if OOM with the default 128
# ============================================================================

# NOTE: deliberately NOT using 'set -u' (nounset). macOS ships bash 3.2 which
# handles empty arrays under nounset inconsistently with bash 4+/5+. -e -o
# pipefail still catches command failures and broken pipelines, which is all
# we need here. Works the same on Ubuntu (bash 5).
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V3_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${V3_ROOT}"
echo "[v4] V3_ROOT=${V3_ROOT}"

# --- tunables (defaults: V4 film main run) ---
MODEL="${MODEL:-film}"
HOLDOUT="${HOLDOUT:-B1,Tregs,plasma,mDC1}"
VAL_HOLDOUT_CELLS="${VAL_HOLDOUT_CELLS:-mDC1}"
LOSS_KIND="${LOSS_KIND:-focal_balanced}"
FOCAL_GAMMA="${FOCAL_GAMMA:-2.0}"
MAX_EPOCHS="${MAX_EPOCHS:-12}"
BATCH_SIZE="${BATCH_SIZE:-128}"   # V4 default lowered from 256 to 128 (bigger model + 4 kb window)
NUM_WORKERS="${NUM_WORKERS:-8}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-42}"
LR="${LR:-1e-3}"
WD="${WD:-1e-4}"
CELL_EMBED_DIM="${CELL_EMBED_DIM:-32}"
PATIENCE="${PATIENCE:-3}"
SEED_RULE="${SEED_RULE:-any_7mer}"
UTR_HOLDOUT_FRAC="${UTR_HOLDOUT_FRAC:-0.20}"
MIRNA_HOLDOUT_FRAC="${MIRNA_HOLDOUT_FRAC:-0.20}"
LOG_EVERY="${LOG_EVERY:-200}"
MAX_CHUNKS="${MAX_CHUNKS:-0}"

# V4-specific knobs
UTR_WINDOW_LEN="${UTR_WINDOW_LEN:-4000}"
AUX_SEED_WEIGHT="${AUX_SEED_WEIGHT:-0.3}"

# --- mode flags ---
MIXED_ONLY="${MIXED_ONLY:-0}"
NO_AUX="${NO_AUX:-0}"
NO_CONVNEXT="${NO_CONVNEXT:-0}"

# --- build run tag automatically (informative + comparable across runs) ---
TAG_PREFIX="v4_${MODEL}"
[[ "${MIXED_ONLY}" == "1" ]] && TAG_PREFIX="${TAG_PREFIX}_mixed"
[[ "${NO_AUX}"     == "1" ]] && TAG_PREFIX="${TAG_PREFIX}_noaux"
[[ "${NO_CONVNEXT}" == "1" ]] && TAG_PREFIX="${TAG_PREFIX}_nocnxt"
TAG="${TAG:-${TAG_PREFIX}_seed${SEED}}"

# --- sanity: cache must already exist ---
if [[ ! -f "cache/splits/train_rows.npz" ]]; then
    echo "[error] cache/splits/train_rows.npz not found."
    echo "[error] Please run ./script/run_all.sh first to build the cache."
    exit 1
fi
echo "[v4] cache present; will reuse it"

mkdir -p runs

# --- assemble args ---
EXTRA_FLAGS=()
[[ "${NO_AUX}"      == "1" ]] && EXTRA_FLAGS+=( --no_aux_seed_head )
[[ "${NO_CONVNEXT}" == "1" ]] && EXTRA_FLAGS+=( --no_convnext )
[[ "${MIXED_ONLY}"  == "1" ]] && EXTRA_FLAGS+=( --train_mixed_only )

echo ""
echo "============================================================"
echo "[v4] TRAINING V4 -- variant=${MODEL}  tag=${TAG}"
echo "      utr_window=${UTR_WINDOW_LEN}  aux_weight=${AUX_SEED_WEIGHT}"
echo "      batch=${BATCH_SIZE}  max_epochs=${MAX_EPOCHS}  patience=${PATIENCE}"
if [[ ${#EXTRA_FLAGS[@]} -gt 0 ]]; then
    echo "      flags: ${EXTRA_FLAGS[*]}"
else
    echo "      flags: <none>"
fi
echo "============================================================"

python script/train_main_model_v4.py \
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
    --max_chunks "${MAX_CHUNKS}" \
    --utr_window_len "${UTR_WINDOW_LEN}" \
    --aux_seed_weight "${AUX_SEED_WEIGHT}" \
    "${EXTRA_FLAGS[@]}"

echo ""
echo "============================================================"
echo "[v4] DONE -- report: ${V3_ROOT}/runs/${TAG}/final_report.json"
echo "[v4] compare with V3:"
echo "    python script/summarize_results.py --include_tags ${TAG} main_${MODEL}_seed${SEED} diag_mixed_${MODEL}_seed${SEED}"
echo "============================================================"
