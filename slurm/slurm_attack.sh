#!/usr/bin/env bash
#SBATCH --job-name=sar-atr-attack
#SBATCH --output=slurm/logs/attack_%A_%a.out
#SBATCH --error=slurm/logs/attack_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-134                # 3 models x 5 seeds x 9 (attack, eps) = 135 tasks
# --------------------------------------------------------------------------
# SAR ATR adversarial evaluation array: one (model, seed, attack, eps) per task.
# Typical submission (after training array finishes):
#
#     sbatch --dependency=afterok:$TRAIN_JOB_ID slurm/slurm_attack.sh
#
# The 9 (attack, eps) combinations are:
#   fgsm x {0.01, 0.02, 0.05, 0.10}     -> 4 tasks per (model, seed)
#   pgd  x {0.01, 0.02, 0.05, 0.10}     -> 4 tasks
#   cw   x {1.0}  (cw_c; epsilon is nominal) -> 1 task
# --------------------------------------------------------------------------

set -euo pipefail
mkdir -p slurm/logs

# shellcheck disable=SC1091
source slurm/env_setup.sh

MODELS=(resnet50 efficientnet_b3 vit_b_16)
SEEDS=(0 1 2 3 4)
ATTACKS=(fgsm fgsm fgsm fgsm pgd pgd pgd pgd cw)
EPSILONS=(0.01 0.02 0.05 0.10 0.01 0.02 0.05 0.10 0.0)

NUM_MODELS=${#MODELS[@]}
NUM_SEEDS=${#SEEDS[@]}
NUM_ATTACKS=${#ATTACKS[@]}
TOTAL=$((NUM_MODELS * NUM_SEEDS * NUM_ATTACKS))

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID >= TOTAL )); then
    echo "Task id $TASK_ID outside range 0..$((TOTAL-1))"; exit 1
fi

# Index math: outermost loop = model, middle = seed, innermost = (attack, eps).
ATTACK_IDX=$(( TASK_ID % NUM_ATTACKS ))
Q1=$((            TASK_ID / NUM_ATTACKS ))
SEED_IDX=$((      Q1       % NUM_SEEDS ))
MODEL_IDX=$((     Q1       / NUM_SEEDS ))

MODEL="${MODELS[$MODEL_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
ATTACK="${ATTACKS[$ATTACK_IDX]}"
EPSILON="${EPSILONS[$ATTACK_IDX]}"

DATASET="${SAR_ATR_DATASET:-atrnet_star}"
DATA_DIR="${SAR_ATR_DATA_DIR:-${SAR_ATR_DATA_ROOT}/atrnet_star}"
CKPT="${SAR_ATR_CHECKPOINT_DIR}/${DATASET}/${MODEL}/seed_${SEED}/best_model.pth"

# Keep per-array-task CSVs so parallel writes don't race. Aggregate with
# slurm/aggregate_results.sh after the array completes.
RESULTS_CSV="${SAR_ATR_RESULTS_DIR}/attacks/${DATASET}/${MODEL}/seed_${SEED}/${ATTACK}_eps${EPSILON}.csv"

# CW ignores --epsilon in practice; we still pass a placeholder so the CSV
# schema is uniform across rows.
BATCH_SIZE=32
if [[ "${MODEL}" == "vit_b_16" ]]; then
    # Adversarial generation holds activations + gradients; lower batch here.
    BATCH_SIZE=16
fi

echo "[attack] task=${TASK_ID} model=${MODEL} seed=${SEED} attack=${ATTACK} eps=${EPSILON}"
echo "[attack] checkpoint=${CKPT}"
echo "[attack] results_csv=${RESULTS_CSV}"
nvidia-smi || true

srun sar-atr-attack \
    --dataset "${DATASET}" \
    --model "${MODEL}" \
    --seed "${SEED}" \
    --attack_type "${ATTACK}" \
    --epsilon "${EPSILON}" \
    --checkpoint_path "${CKPT}" \
    --data_dir "${DATA_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --results_csv "${RESULTS_CSV}"
