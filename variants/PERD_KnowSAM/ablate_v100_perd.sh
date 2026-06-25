#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

BASE_SNAPSHOT_PATH="${BASE_SNAPSHOT_PATH:-./Results/PERD_KnowSAM_Ablations}"
DATA_PATH="${DATA_PATH:-./SampleData}"
DATASET="${DATASET:-/260513_data_label1}"
MAX_ITERATIONS="${MAX_ITERATIONS:-50000}"
VAL_INTERVAL="${VAL_INTERVAL:-200}"

run_ablation() {
  local name="$1"
  shift
  echo "Running PERD ablation: ${name}"
  env \
    DATA_PATH="${DATA_PATH}" \
    DATASET="${DATASET}" \
    MAX_ITERATIONS="${MAX_ITERATIONS}" \
    VAL_INTERVAL="${VAL_INTERVAL}" \
    SNAPSHOT_PATH="${BASE_SNAPSHOT_PATH}/${name}" \
    "$@" \
    bash ./variants/PERD_KnowSAM/train_v100_perd.sh
}

run_ablation "baseline_knowsam" PERD_ENABLED=0
run_ablation "perd_full" PERD_ENABLED=1 PERD_BASELINE=perd PERD_ATTENUATION_MODE=boundary
run_ablation "pc_only" PERD_ENABLED=1 PERD_DISABLE_ED=1 PERD_BASELINE=perd
run_ablation "ed_only" PERD_ENABLED=1 PERD_DISABLE_PC=1 PERD_BASELINE=perd
run_ablation "prompt_dose_only" PERD_ENABLED=1 PERD_NO_ATTENUATION=1 PERD_BASELINE=perd
run_ablation "prompt_ensemble_equal_query" PERD_ENABLED=1 PERD_BASELINE=prompt_ensemble
run_ablation "random_tube_attenuation" PERD_ENABLED=1 PERD_ATTENUATION_MODE=random
run_ablation "interior_attenuation" PERD_ENABLED=1 PERD_ATTENUATION_MODE=interior

echo "All ablations finished under ${BASE_SNAPSHOT_PATH}"
