#!/usr/bin/env bash
#SBATCH --job-name=sar-atr-train
#SBATCH --output=slurm/logs/train_%A_%a.out
#SBATCH --error=slurm/logs/train_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-14                # 3 models x 5 seeds = 15 tasks (indices 0..14)
# --------------------------------------------------------------------------
# SAR ATR training array: one (model, seed) combination per task.
# Submit with `sbatch slurm/slurm_train.sh` from the project root.
# --------------------------------------------------------------------------

set -euo pipefail

# Ensure log dir exists on first call even if `#SBATCH --output` creates it.
mkdir -p slurm/logs

# shellcheck disable=SC1091
source slurm/env_setup.sh

# ---- task -> (model, seed) mapping ------------------------------------------
MODELS=(resnet50 efficientnet_b3 vit_b_16)
SEEDS=(0 1 2 3 4)

NUM_MODELS=${#MODELS[@]}
NUM_SEEDS=${#SEEDS[@]}
TOTAL=$((NUM_MODELS * NUM_SEEDS))

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID >= TOTAL )); then
    echo "Task id $TASK_ID outside range 0..$((TOTAL-1))"; exit 1
fi

MODEL_IDX=$(( TASK_ID / NUM_SEEDS ))
SEED_IDX=$((  TASK_ID % NUM_SEEDS ))
MODEL="${MODELS[$MODEL_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

# ViT-B is memory hungry; halve the batch size and double grad accumulation
# so an effective batch of 64 still fits in a single L40 (48GB).
BATCH_SIZE=64
GRAD_ACCUM=1
if [[ "${MODEL}" == "vit_b_16" ]]; then
    BATCH_SIZE=32
    GRAD_ACCUM=2
fi

DATASET="${SAR_ATR_DATASET:-atrnet_star}"
DATA_DIR="${SAR_ATR_DATA_DIR:-${SAR_ATR_DATA_ROOT}/atrnet_star}"
EPOCHS="${SAR_ATR_EPOCHS:-50}"
SUMMARY_CSV="${SAR_ATR_RESULTS_DIR}/train_summary.csv"

echo "[train] task=${TASK_ID} model=${MODEL} seed=${SEED} dataset=${DATASET}"
echo "[train] batch_size=${BATCH_SIZE} grad_accum=${GRAD_ACCUM} epochs=${EPOCHS}"
echo "[train] data_dir=${DATA_DIR}"
nvidia-smi || true

srun python "${SAR_ATR_PROJECT_DIR}/train.py" \
    --dataset "${DATASET}" \
    --model "${MODEL}" \
    --seed "${SEED}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --grad_accum_steps "${GRAD_ACCUM}" \
    --data_dir "${DATA_DIR}" \
    --summary_csv "${SUMMARY_CSV}" \
    --resume
