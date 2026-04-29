#!/usr/bin/env bash
# Merge the per-task attack CSVs written by the SLURM array into a single
# `attack_results.csv` for downstream analysis.
#
# Usage (from project root):
#     ./slurm/aggregate_results.sh [<results_root>]

set -euo pipefail

RESULTS_ROOT="${1:-${SAR_ATR_RESULTS_DIR:-results}/attacks}"
OUTPUT="${RESULTS_ROOT%/attacks}/attack_results.csv"

if [[ ! -d "${RESULTS_ROOT}" ]]; then
    echo "No results directory at ${RESULTS_ROOT}." >&2
    exit 1
fi

mapfile -t CSVS < <(find "${RESULTS_ROOT}" -type f -name "*.csv" | sort)
if (( ${#CSVS[@]} == 0 )); then
    echo "No per-task CSVs found under ${RESULTS_ROOT}." >&2
    exit 1
fi

# Use the header of the first file, then append the data rows of every file.
head -n 1 "${CSVS[0]}" > "${OUTPUT}"
for f in "${CSVS[@]}"; do
    tail -n +2 "${f}" >> "${OUTPUT}"
done

echo "Aggregated ${#CSVS[@]} CSV(s) into ${OUTPUT}"
wc -l "${OUTPUT}"
