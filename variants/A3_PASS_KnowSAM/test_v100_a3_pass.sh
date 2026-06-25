#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_PATH="${DATA_PATH:-./SampleData}"
DATASET="${DATASET:-/260513_data_label1}"
SPLIT="${SPLIT:-test}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SNAPSHOT_PATH="${SNAPSHOT_PATH:-./Results/A3_PASS_KnowSAM_V100_label1_106_117_13_13}"
PASS_MODEL_PATH="${PASS_MODEL_PATH:-${SNAPSHOT_PATH}/fold_0/PASS_best_model.pth}"
SAVE_DIR="${SAVE_DIR:-${SNAPSHOT_PATH}/fold_0/prediction_${SPLIT}}"

PASS_STATE_SIZE="${PASS_STATE_SIZE:-64}"
PASS_STATE_DIM="${PASS_STATE_DIM:-64}"
PASS_BASE_CHANNELS="${PASS_BASE_CHANNELS:-32}"

if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=8
fi

if [[ ! -f "${PASS_MODEL_PATH}" ]]; then
  echo "Missing PASS checkpoint: ${PASS_MODEL_PATH}" >&2
  exit 1
fi

mkdir -p "${SAVE_DIR}"

echo "Starting A3-PASS-KnowSAM evaluation with:"
echo "  PYTHON_BIN=${PYTHON_BIN}"
echo "  DATA_PATH=${DATA_PATH}"
echo "  DATASET=${DATASET}"
echo "  SPLIT=${SPLIT}"
echo "  IMAGE_SIZE=${IMAGE_SIZE}"
echo "  PASS_MODEL_PATH=${PASS_MODEL_PATH}"
echo "  SAVE_DIR=${SAVE_DIR}"
echo "  NUM_WORKERS=${NUM_WORKERS}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

"${PYTHON_BIN}" ./variants/A3_PASS_KnowSAM/prediction_a3_pass.py \
  --data_path "${DATA_PATH}" \
  --dataset "${DATASET}" \
  --split "${SPLIT}" \
  --image_size "${IMAGE_SIZE}" \
  --PASS_model_path "${PASS_MODEL_PATH}" \
  --save_dir "${SAVE_DIR}" \
  --num_workers "${NUM_WORKERS}" \
  --pass_state_size "${PASS_STATE_SIZE}" \
  --pass_state_dim "${PASS_STATE_DIM}" \
  --pass_base_channels "${PASS_BASE_CHANNELS}" \
  "$@"

echo "Evaluation finished. Outputs:"
echo "  ${SAVE_DIR}"
echo "  ${SAVE_DIR}/prediction.log"
echo "  ${SAVE_DIR}/monitor"
