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
SNAPSHOT_PATH="${SNAPSHOT_PATH:-./Results/PERD_KnowSAM_V100_label1_106_117_13_13}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-./sam_vit_b_01ec64.pth}"

PERD_ENABLED="${PERD_ENABLED:-1}"
PERD_DELTA_LEVELS="${PERD_DELTA_LEVELS:--4,-2,0,2,4}"
PERD_ATTENUATION_MODE="${PERD_ATTENUATION_MODE:-boundary}"
PERD_BASELINE="${PERD_BASELINE:-perd}"
PERD_DISABLE_ED="${PERD_DISABLE_ED:-0}"
PERD_DISABLE_PC="${PERD_DISABLE_PC:-0}"
PERD_NO_ATTENUATION="${PERD_NO_ATTENUATION:-0}"

if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=8
fi

mkdir -p "${SNAPSHOT_PATH}"

echo "Starting PERD-KnowSAM training with:"
echo "  DATA_PATH=${DATA_PATH}"
echo "  DATASET=${DATASET}"
echo "  BATCH_SIZE=${BATCH_SIZE}"
echo "  LABELED_BS=${LABELED_BS}"
echo "  MAX_ITERATIONS=${MAX_ITERATIONS}"
echo "  SNAPSHOT_PATH=${SNAPSHOT_PATH}"
echo "  PERD_BASELINE=${PERD_BASELINE}"
echo "  PERD_ATTENUATION_MODE=${PERD_ATTENUATION_MODE}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

"${PYTHON_BIN}" ./variants/PERD_KnowSAM/train_semi_SAM_perd.py \
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
  --perd_enabled "${PERD_ENABLED}" \
  --perd_delta_levels "${PERD_DELTA_LEVELS}" \
  --perd_attenuation_mode "${PERD_ATTENUATION_MODE}" \
  --perd_baseline "${PERD_BASELINE}" \
  --perd_disable_ed "${PERD_DISABLE_ED}" \
  --perd_disable_pc "${PERD_DISABLE_PC}" \
  --perd_no_attenuation "${PERD_NO_ATTENUATION}" \
  "$@"

echo "Training finished. Outputs:"
echo "  ${SNAPSHOT_PATH}"
