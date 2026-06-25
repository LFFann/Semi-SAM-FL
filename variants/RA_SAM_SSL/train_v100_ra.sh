#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_PATH="${DATA_PATH:-./SampleData}"
DATASET="${DATASET:-/260513_data_label1}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
BATCH_SIZE="${BATCH_SIZE:-24}"
LABELED_BS="${LABELED_BS:-12}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-2}"
MAX_ITERATIONS="${MAX_ITERATIONS:-50000}"
MIXED_ITERATIONS="${MIXED_ITERATIONS:-12000}"
VAL_INTERVAL="${VAL_INTERVAL:-200}"
SNAPSHOT_PATH="${SNAPSHOT_PATH:-./Results/RA_SAM_SSL_V100_label1_106_117_13_13}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-./sam_vit_b_01ec64.pth}"

RA_ENABLED="${RA_ENABLED:-1}"
RA_DELTA_LEVELS="${RA_DELTA_LEVELS:--4,-2,0,2,4}"
RA_INTERVENTION_MODE="${RA_INTERVENTION_MODE:-boundary}"
RA_BASELINE="${RA_BASELINE:-response_audit}"
RA_DISABLE_ESR="${RA_DISABLE_ESR:-0}"
RA_DISABLE_PROMPT="${RA_DISABLE_PROMPT:-0}"
RA_NO_INTERVENTION="${RA_NO_INTERVENTION:-0}"
RA_ENFORCE_INTERVENTION_VALIDITY="${RA_ENFORCE_INTERVENTION_VALIDITY:-1}"

if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=8
fi

mkdir -p "${SNAPSHOT_PATH}"

echo "Starting RA-SAM-SSL training with:"
echo "  DATA_PATH=${DATA_PATH}"
echo "  DATASET=${DATASET}"
echo "  BATCH_SIZE=${BATCH_SIZE}"
echo "  LABELED_BS=${LABELED_BS}"
echo "  MAX_ITERATIONS=${MAX_ITERATIONS}"
echo "  SNAPSHOT_PATH=${SNAPSHOT_PATH}"
echo "  RA_BASELINE=${RA_BASELINE}"
echo "  RA_INTERVENTION_MODE=${RA_INTERVENTION_MODE}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

"${PYTHON_BIN}" ./variants/RA_SAM_SSL/train_semi_SAM_ra.py \
  --data_path "${DATA_PATH}" \
  --dataset "${DATASET}" \
  --image_size "${IMAGE_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --labeled_bs "${LABELED_BS}" \
  --num_workers "${NUM_WORKERS}" \
  --val_num_workers "${VAL_NUM_WORKERS}" \
  --max_iterations "${MAX_ITERATIONS}" \
  --mixed_iterations "${MIXED_ITERATIONS}" \
  --val_interval "${VAL_INTERVAL}" \
  --snapshot_path "${SNAPSHOT_PATH}" \
  --sam_checkpoint "${SAM_CHECKPOINT}" \
  --ra_enabled "${RA_ENABLED}" \
  --ra_delta_levels "${RA_DELTA_LEVELS}" \
  --ra_intervention_mode "${RA_INTERVENTION_MODE}" \
  --ra_baseline "${RA_BASELINE}" \
  --ra_disable_esr "${RA_DISABLE_ESR}" \
  --ra_disable_prompt "${RA_DISABLE_PROMPT}" \
  --ra_no_intervention "${RA_NO_INTERVENTION}" \
  --ra_enforce_intervention_validity "${RA_ENFORCE_INTERVENTION_VALIDITY}" \
  "$@"

echo "Training finished. Outputs:"
echo "  ${SNAPSHOT_PATH}"

