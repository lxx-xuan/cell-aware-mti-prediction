#!/usr/bin/env bash
# ============================================================================
# V3 — Cell-aware MTI full experiment runner.
#
# Workflow:
#   1. Build the cache ONCE on all 26 chunks with a fixed (manual) holdout.
#   2. Train 4 model variants (full / no_cell / no_attn / cell_only),
#      each reusing the same cache.
#   3. Summarize all runs into a single comparison table.
#
# Defaults are tuned for an Ubuntu box with an RTX-class GPU and >=16 GB RAM.
# Override anything by passing it in via environment variables, e.g.:
#   HOLDOUT="B1,Tregs,plasma" MAX_EPOCHS=15 BATCH_SIZE=256 ./run_all.sh
#
# To run a single variant for debugging instead of the full sweep:
#   ONLY="full" ./run_all.sh
# ============================================================================

set -euo pipefail

# --- locate V3_ROOT (the parent of this script's directory) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V3_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${V3_ROOT}"
echo "[run_all] V3_ROOT=${V3_ROOT}"

# --- tunables ---
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
MAX_CHUNKS="${MAX_CHUNKS:-0}"   # 0 = use all chunks
ONLY="${ONLY:-}"                 # comma-separated variant subset (empty = all 4)
LOG_EVERY="${LOG_EVERY:-200}"
SEED_RULE="${SEED_RULE:-any_7mer}"   # 7mer-m8 / 7mer-A1 / 6mer / 8mer / any_7mer / any
UTR_HOLDOUT_FRAC="${UTR_HOLDOUT_FRAC:-0.20}"
MIRNA_HOLDOUT_FRAC="${MIRNA_HOLDOUT_FRAC:-0.20}"

# variants to sweep
# 'film' is the main model (FiLM cell-conditioned modulation).
# 'full' kept as baseline to compare cell-injection-via-concat vs cell-injection-via-FiLM.
ALL_VARIANTS="film full no_cell no_attn cell_only"
if [[ -n "${ONLY}" ]]; then
    VARIANTS="$(echo "${ONLY}" | tr ',' ' ')"
else
    VARIANTS="${ALL_VARIANTS}"
fi

echo "[run_all] holdout_cell_types=${HOLDOUT}"
echo "[run_all] seed_rule=${SEED_RULE} utr_holdout=${UTR_HOLDOUT_FRAC} mirna_holdout=${MIRNA_HOLDOUT_FRAC} val_holdout_cells=${VAL_HOLDOUT_CELLS}"
echo "[run_all] max_epochs=${MAX_EPOCHS} batch_size=${BATCH_SIZE} num_workers=${NUM_WORKERS} device=${DEVICE}"
echo "[run_all] variants: ${VARIANTS}"
echo "[run_all] max_chunks=${MAX_CHUNKS} (0 = all)"

mkdir -p runs

# --- Step 1: rebuild cache once with fixed holdout ---
echo ""
echo "============================================================"
echo "[step 1] BUILDING CACHE (rebuild + max_epochs=0)"
echo "============================================================"
python script/train_main_model.py \
    --run_tag _cache_build_only \
    --rebuild_cache \
    --max_epochs 0 \
    --holdout_cell_types "${HOLDOUT}" \
    --seed "${SEED}" \
    --seed_rule "${SEED_RULE}" \
    --utr_holdout_frac "${UTR_HOLDOUT_FRAC}" \
    --mirna_holdout_frac "${MIRNA_HOLDOUT_FRAC}" \
    --val_holdout_cell_types "${VAL_HOLDOUT_CELLS}" \
    --max_chunks "${MAX_CHUNKS}"

# --- Step 2: train each variant, reusing the shared cache ---
for KIND in ${VARIANTS}; do
    TAG="main_${KIND}_seed${SEED}"
    echo ""
    echo "============================================================"
    echo "[step 2] TRAINING variant=${KIND} -> run_tag=${TAG}"
    echo "============================================================"
    python script/train_main_model.py \
        --run_tag "${TAG}" \
        --model "${KIND}" \
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
done

# --- Step 3: aggregate results ---
echo ""
echo "============================================================"
echo "[step 3] SUMMARIZING"
echo "============================================================"
python script/summarize_results.py --include_tags "main_"

# --- Step 4: leakage sanity check ---
echo ""
echo "============================================================"
echo "[step 4] LEAKAGE SANITY CHECK"
echo "============================================================"
python script/diagnose_leakage.py

echo ""
echo "[run_all] all done."
echo "[run_all] tables: ${V3_ROOT}/runs/_summary/"
echo "[run_all] per-run reports: ${V3_ROOT}/runs/main_*"
