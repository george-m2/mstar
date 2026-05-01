#!/usr/bin/env bash
# Convenience launcher: submits the training array and chains the attack
# array with --dependency=afterok:$TRAIN_JOB_ID so attacks start automatically
# once all training tasks complete successfully.
#
# Usage (from project root):
#     ./slurm/submit.sh                # full pipeline
#     ./slurm/submit.sh --attacks-only  # skip training, attack existing ckpts
#     ./slurm/submit.sh --train-only    # only training

set -euo pipefail

MODE="${1:-full}"

case "${MODE}" in
    --attacks-only|attacks-only)
        echo "Submitting attack array only..."
        sbatch slurm/slurm_attack.sh
        ;;
    --train-only|train-only)
        echo "Submitting training array only..."
        sbatch slurm/slurm_train.sh
        ;;
    --full|full|"")
        TRAIN_JOB=$(sbatch --parsable slurm/slurm_train.sh)
        echo "Submitted training array as job ${TRAIN_JOB}"
        ATTACK_JOB=$(sbatch --parsable --dependency="afterok:${TRAIN_JOB}" slurm/slurm_attack.sh)
        echo "Submitted attack array as job ${ATTACK_JOB} (waits for ${TRAIN_JOB})"
        ;;
    *)
        echo "Usage: $0 [--full | --train-only | --attacks-only]"
        exit 1
        ;;
esac
