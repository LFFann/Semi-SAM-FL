#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

BASE_SNAPSHOT_PATH="${BASE_SNAPSHOT_PATH:-./Results/RA_SAM_SSL_Ablations}"
DATA_PATH="${DATA_PATH:-./SampleData}"
DATASET="${DATASET:-/260513_data_label1}"
MAX_ITERATIONS="${MAX_ITERATIONS:-50000}"
VAL_INTERVAL="${VAL_INTERVAL:-200}"

run_ablation() {
  local name="$1"
  shift
  echo "Running RA-SAM-SSL ablation: ${name}"
  env \
    DATA_PATH="${DATA_PATH}" \
    DATASET="${DATASET}" \
    MAX_ITERATIONS="${MAX_ITERATIONS}" \
    VAL_INTERVAL="${VAL_INTERVAL}" \
    SNAPSHOT_PATH="${BASE_SNAPSHOT_PATH}/${name}" \
    "$@" \
    bash ./variants/RA_SAM_SSL/train_v100_ra.sh
}

run_ablation "baseline_knowsam" RA_ENABLED=0
run_ablation "ra_full" RA_ENABLED=1 RA_BASELINE=response_audit RA_INTERVENTION_MODE=boundary
run_ablation "prompt_response_only" RA_ENABLED=1 RA_DISABLE_ESR=1 RA_BASELINE=response_audit
run_ablation "evidence_response_only" RA_ENABLED=1 RA_DISABLE_PROMPT=1 RA_BASELINE=response_audit
run_ablation "prompt_dose_only" RA_ENABLED=1 RA_NO_INTERVENTION=1 RA_ENFORCE_INTERVENTION_VALIDITY=0 RA_BASELINE=response_audit
run_ablation "prompt_ensemble_equal_query" RA_ENABLED=1 RA_BASELINE=prompt_ensemble
run_ablation "random_tube_intervention" RA_ENABLED=1 RA_INTERVENTION_MODE=random
run_ablation "interior_intervention" RA_ENABLED=1 RA_INTERVENTION_MODE=interior

echo "All ablations finished under ${BASE_SNAPSHOT_PATH}"

