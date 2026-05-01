#!/usr/bin/env bash
# Environment setup sourced by every SLURM job. Keeps cluster-specific
# path/module logic in one place so the batch scripts stay portable.
# Override any of these before `sbatch` to adapt to a different site.

set -euo pipefail

# ---- cluster paths (override per-site) --------------------------------------
export SAR_ATR_PROJECT_DIR="${SAR_ATR_PROJECT_DIR:-$PWD}"
# Parallel/scratch filesystem holding the ATRNet-STAR / MSTAR data.
export SAR_ATR_DATA_ROOT="${SAR_ATR_DATA_ROOT:-/scratch/${USER}/datasets}"
# Persistent checkpoint + results locations (NFS / project dir is fine).
export SAR_ATR_CHECKPOINT_DIR="${SAR_ATR_CHECKPOINT_DIR:-${SAR_ATR_PROJECT_DIR}/checkpoints}"
export SAR_ATR_RESULTS_DIR="${SAR_ATR_RESULTS_DIR:-${SAR_ATR_PROJECT_DIR}/results}"
# Share torchvision pretrained weights across jobs to avoid re-downloads.
export SAR_ATR_MODEL_CACHE_DIR="${SAR_ATR_MODEL_CACHE_DIR:-${SAR_ATR_PROJECT_DIR}/.torch}"
export TORCH_HOME="${SAR_ATR_MODEL_CACHE_DIR}"

# ---- module system ----------------------------------------------------------
# Uncomment and edit for your cluster's module environment. Examples:
# module purge
# module load cuda/12.4
# module load python/3.12

# ---- uv-managed virtualenv --------------------------------------------------
# Create once on a login node:
#   module load python/3.12 && uv sync
# Subsequent jobs just activate the resulting .venv.
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ -d "${SAR_ATR_PROJECT_DIR}/.venv" ]]; then
        # shellcheck disable=SC1091
        source "${SAR_ATR_PROJECT_DIR}/.venv/bin/activate"
    else
        echo "[env_setup] WARNING: no .venv found under ${SAR_ATR_PROJECT_DIR}." >&2
        echo "[env_setup] Run 'uv sync' on a login node before submitting jobs." >&2
    fi
fi

# Prefer the local package source over any installed copy.
export PYTHONPATH="${SAR_ATR_PROJECT_DIR}/src:${PYTHONPATH:-}"

# Avoid oversubscription when dataloader workers spawn.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

echo "[env_setup] project=${SAR_ATR_PROJECT_DIR}"
echo "[env_setup] data=${SAR_ATR_DATA_ROOT}"
echo "[env_setup] checkpoints=${SAR_ATR_CHECKPOINT_DIR}"
echo "[env_setup] results=${SAR_ATR_RESULTS_DIR}"
echo "[env_setup] python=$(command -v python) torch=$(python -c 'import torch;print(torch.__version__)' 2>/dev/null || echo missing)"
