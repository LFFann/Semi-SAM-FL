#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_PATH="${DATA_PATH:-./SampleData}"
DATASET="${DATASET:-/260513_data_labeled30pct}"
SPLIT="${SPLIT:-test}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
NUM_CLASSES="${NUM_CLASSES:-3}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SNAPSHOT_PATH="${SNAPSHOT_PATH:-./Results/SRG-SAM_Lite_V100_260513_data_labeled30pct_from_scratch}"
MODEL_PATH="${MODEL_PATH:-${SNAPSHOT_PATH}/SRG_SAM_Lite_deploy_SGDL_best_model.pth}"
DPG_MODEL_PATH="${DPG_MODEL_PATH:-${SNAPSHOT_PATH}/SRG_SAM_Lite_deploy_DPG_best_model.pth}"
RELIABILITY_MODEL_PATH="${RELIABILITY_MODEL_PATH:-${SNAPSHOT_PATH}/SRG_SAM_Lite_deploy_reliability_best_model.pth}"
PREDICTION_MODE="${PREDICTION_MODE:-deploy}"
USE_DPG_DEPLOY="${USE_DPG_DEPLOY:-true}"
USE_RELIABILITY_HEAD="${USE_RELIABILITY_HEAD:-true}"
SAVE_DIR="${SAVE_DIR:-${SNAPSHOT_PATH}/prediction_${SPLIT}}"

if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=8
fi

REQUIRED_FILES=("${MODEL_PATH}")
if [[ "${PREDICTION_MODE}" == "graph" || ( "${PREDICTION_MODE}" != "student" && "${USE_DPG_DEPLOY}" != "false" && "${USE_DPG_DEPLOY}" != "0" ) ]]; then
  REQUIRED_FILES+=("${DPG_MODEL_PATH}")
fi
if [[ "${PREDICTION_MODE}" == "deploy" && "${USE_RELIABILITY_HEAD}" != "false" && "${USE_RELIABILITY_HEAD}" != "0" ]]; then
  REQUIRED_FILES+=("${RELIABILITY_MODEL_PATH}")
fi
for required_file in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Missing required file: ${required_file}" >&2
    exit 1
  fi
done

mkdir -p "${SAVE_DIR}"

echo "Starting SRG-SAM multiclass evaluation with:"
echo "  PYTHON_BIN=${PYTHON_BIN}"
echo "  DATA_PATH=${DATA_PATH}"
echo "  DATASET=${DATASET}"
echo "  SPLIT=${SPLIT}"
echo "  NUM_CLASSES=${NUM_CLASSES}"
echo "  MODEL_PATH=${MODEL_PATH}"
echo "  DPG_MODEL_PATH=${DPG_MODEL_PATH}"
echo "  RELIABILITY_MODEL_PATH=${RELIABILITY_MODEL_PATH}"
echo "  PREDICTION_MODE=${PREDICTION_MODE}"
echo "  USE_DPG_DEPLOY=${USE_DPG_DEPLOY}"
echo "  USE_RELIABILITY_HEAD=${USE_RELIABILITY_HEAD}"
echo "  SAVE_DIR=${SAVE_DIR}"

"${PYTHON_BIN}" ./variants/Multiclass_KnowSAM/prediction_multiclass.py \
  --data_path "${DATA_PATH}" \
  --dataset "${DATASET}" \
  --split "${SPLIT}" \
  --num_classes "${NUM_CLASSES}" \
  --image_size "${IMAGE_SIZE}" \
  --SGDL_model_path "${MODEL_PATH}" \
  --DPG_model_path "${DPG_MODEL_PATH}" \
  --reliability_model_path "${RELIABILITY_MODEL_PATH}" \
  --prediction_mode "${PREDICTION_MODE}" \
  --use_dpg_deploy "${USE_DPG_DEPLOY}" \
  --use_reliability_head "${USE_RELIABILITY_HEAD}" \
  --save_dir "${SAVE_DIR}" \
  --num_workers "${NUM_WORKERS}" \
  "$@"

echo "Evaluation finished. Outputs:"
echo "  ${SAVE_DIR}"
