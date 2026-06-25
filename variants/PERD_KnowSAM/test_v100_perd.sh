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
SNAPSHOT_PATH="${SNAPSHOT_PATH:-./Results/PERD_KnowSAM_V100_label1_106_117_13_13}"
SGDL_MODEL_PATH="${SGDL_MODEL_PATH:-${SNAPSHOT_PATH}/SGDL_best_model.pth}"
SAM_MODEL_PATH="${SAM_MODEL_PATH:-${SNAPSHOT_PATH}/sam_best_model.pth}"
SAVE_DIR="${SAVE_DIR:-${SNAPSHOT_PATH}/prediction_${SPLIT}}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-./sam_vit_b_01ec64.pth}"

if [[ ! -f "${SGDL_MODEL_PATH}" ]]; then
  SGDL_MODEL_PATH="${SNAPSHOT_PATH}/fold_0/SGDL_best_model.pth"
fi
if [[ ! -f "${SAM_MODEL_PATH}" ]]; then
  SAM_MODEL_PATH="${SNAPSHOT_PATH}/fold_0/sam_best_model.pth"
fi
if [[ ! -f "${SGDL_MODEL_PATH}" ]]; then
  echo "Missing SGDL checkpoint: ${SGDL_MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${SAM_MODEL_PATH}" ]]; then
  echo "Missing SAM checkpoint: ${SAM_MODEL_PATH}" >&2
  exit 1
fi

mkdir -p "${SAVE_DIR}"

"${PYTHON_BIN}" ./variants/PERD_KnowSAM/prediction_perd.py \
  --data_path "${DATA_PATH}" \
  --dataset "${DATASET}" \
  --split "${SPLIT}" \
  --image_size "${IMAGE_SIZE}" \
  --sam_checkpoint "${SAM_CHECKPOINT}" \
  --SGDL_model_path "${SGDL_MODEL_PATH}" \
  --sam_model_path "${SAM_MODEL_PATH}" \
  --save_dir "${SAVE_DIR}" \
  --num_workers "${NUM_WORKERS}" \
  "$@"

echo "Evaluation finished. Outputs:"
echo "  ${SAVE_DIR}"
