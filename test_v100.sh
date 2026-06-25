#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_PATH="${DATA_PATH:-./SampleData}"
DATASET="${DATASET:-/260513_data_label1}"
SPLIT="${SPLIT:-test}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MODEL_PATH="${MODEL_PATH:-./Results/train_260513_data_label1_v100_semi_106_117_13_13/SGDL_best_model.pth}"
SAVE_DIR="${SAVE_DIR:-./Results/train_260513_data_label1_v100_semi_106_117_13_13/prediction_test}"

if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=8
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Missing model checkpoint: ${MODEL_PATH}" >&2
  exit 1
fi

mkdir -p "${SAVE_DIR}"

echo "Starting V100 evaluation with:"
echo "  PYTHON_BIN=${PYTHON_BIN}"
echo "  DATA_PATH=${DATA_PATH}"
echo "  DATASET=${DATASET}"
echo "  SPLIT=${SPLIT}"
echo "  IMAGE_SIZE=${IMAGE_SIZE}"
echo "  MODEL_PATH=${MODEL_PATH}"
echo "  SAVE_DIR=${SAVE_DIR}"
echo "  NUM_WORKERS=${NUM_WORKERS}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

"${PYTHON_BIN}" prediction.py \
  --data_path "${DATA_PATH}" \
  --dataset "${DATASET}" \
  --split "${SPLIT}" \
  --image_size "${IMAGE_SIZE}" \
  --SGDL_model_path "${MODEL_PATH}" \
  --save_dir "${SAVE_DIR}" \
  --num_workers "${NUM_WORKERS}" \
  "$@"

echo "Evaluation finished. Outputs:"
echo "  ${SAVE_DIR}"
echo "  ${SAVE_DIR}/prediction.log"
echo "  ${SAVE_DIR}/monitor"
